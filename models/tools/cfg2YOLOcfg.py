import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn


sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from tools.my_yaml import PrettySafeLoader
from models.common import autopad
from models.yolo import Model
from utils.general import make_divisible, colorstr
from utils.torch_utils import intersect_dicts
from utils.datasets import create_dataloader
import test 

def convert_conv(cin, c2, k, d=1, s=1, g=1, p=None, act='swish', bn=1): # swish equals to silu
    if p is None: p = autopad(k,d=d)
    config = {'batch_normalize': bn,
              'filters': c2,
              'size': k,
              'stride': s,
              'dilation': d,
              'pad': p,
              'groups': g,
              'activation': act
             }
    return ('convolutional', config)

def convert_Focus(cin, cout, k):
    """
    Focus equals to a Conv with stride=2, kernel_size=2
    """
#    conv_layer = convert_conv(cin, cout, 2*k,1,2, g=1, p=autopad(k,d=1)*2)
#    return [conv_layer]
    reorg_layer = ('reorg', {'stride':2})
    conv_layer = convert_conv(cin, cout, k,1,1, g=1)
    return [reorg_layer, conv_layer]

def convert_bottleneck(cin, cout, k, d, e, separable=False, shortcut=True):
    layers = []
    c_ = int(cout*e)
    # conv1
    conv1 = convert_conv(cin, c_, 1,1,1, g=1)
    layers.append(conv1)
    if separable:
      conv2 = convert_conv(c_, c_, k,d,1, g=c_, bn=0, act='linear') # dw-conv
      layers.append(conv2)
      conv2 = convert_conv(c_, cout, 1,1,1, g=1) # pw-conv
      layers.append(conv2)
    else:
      conv2 = convert_conv(c_, cout, k,d,1, g=1)
      layers.append(conv2)
    if shortcut:
      assert(cin==cout)
      shortcut_layer = ('shortcut', {'from': -len(layers)-1, 'activation': 'linear'})
      layers.append(shortcut_layer)
    return layers

def convert_c3(n, cin, c2, ks, ds, shortcut=True, args_dict={}):
    e_c3 = 0.5
    e_bs = args_dict.get('e_bottleneck', 1.)
    separable = args_dict.get('separable', False)

    ks = [ks for _ in range(n)] if isinstance(ks, int) else ks
    ds = [ds for _ in range(n)] if isinstance(ds, int) else ds
    e_bs = [e_bs for _ in range(n)] if isinstance(e_bs, float) else e_bs
    c2 = [c2 for _ in range(n)] if isinstance(c2, int) else c2
    c1out = int(c2[0]*e_c3); c2out = int(c2[-1]*e_c3)  # hidden channels

    layers = []; 
    # conv1
    conv1 = convert_conv(cin, c1out, 1, 1, 1, g=1)
    layers.append(conv1)
    # m
    cin = c1out
    for i in range(n):
      cout = int(c2[i]*e_c3)
      bottleneck_layers = convert_bottleneck(cin, cout, ks[i], ds[i], e_bs[i], separable, shortcut)
      cin = cout
      layers.extend(bottleneck_layers)
    # conv2
    route = ('route', {'layers': [-len(layers)-1]})
    layers.append(route)
    conv2 = convert_conv(cin, c2out, 1,1,1, g=1)
    layers.append(conv2)
    # conv3
    route = ('route', {'layers': [-3, -1]}) # in C3, the order of concat is: m, cv2
    layers.append(route)
    assert(c2out == cout)
    conv3 = convert_conv(cout+c2out, c2[-1], 1,1,1, g=1)
    layers.append(conv3)
    return layers

def convert_SPP(cin, cout, ks=(5,9,13)):
    layers = []
    out_layer_ids = [-1]
    for i, k in enumerate(ks):
      if i > 0: 
        layers.append( ('route', {'layers': [-len(layers)-1]}) )
      layers.append( ('maxpool', {'stride': 1, 'size': k}) )
      out_layer_ids.append(len(layers)-1)
    layers.append( ('route', {'layers': [num-len(layers) for num in out_layer_ids]}) )
    # conv1
    c_ = cin // 2
    conv1 = convert_conv(cin, c_, 1,1,1, g=1)
    conv2 = convert_conv(c_*(len(ks)+1), cout, 1,1,1, g=1)
    layers.insert(0, conv1)
    layers.append(conv2)
    return layers

def convert_FF(cins, cout, route_layers, strides, ks, ds, args_dict={}):
    separable = args_dict.get('separable', False)
    assert(len(route_layers)==len(strides))
    out_layer_ids = []
    layers = []
    for i, (route_layer, s) in enumerate(zip(route_layers, strides)):
      if s<1: up_rate = int(1./s); s = 1
      else: up_rate = None
      layers.append( ('route', {'layers': route_layer}) )
      # conv
      assert(ks[i]>0 and ds[i]>0)
      if separable:
        conv2 = convert_conv(cins[i], cins[i], ks[i],ds[i],s, g=cins[i], bn=0, act='linear') # dw-conv
        layers.append(conv2)
        conv2 = convert_conv(cins[i], cout, 1,1,1, g=1) # pw-conv
        layers.append(conv2)
      else:
        conv_layer = convert_conv(cins[i], cout, ks[i], ds[i], s, g=1)
        layers.append(conv_layer)
      # up-sampling
      if up_rate is not None:
        layers.append( ('upsample', {'stride': up_rate}) )
      out_layer_ids.append(len(layers)-1)
    # sum 
    layers.append( ('shortcut', {'from': [x-len(layers) for x in out_layer_ids[:-1]], 'activation': 'linear'}) )
    return layers
      
def convert_Detect(route_layers, nc, anchors, cins):
    no = nc + 5
    layers = []
    total_anchors = []
    for anchor in anchors: total_anchors.extend(anchor)
    mask_start = 0
    for i, route_layer in enumerate(route_layers):
      layers.append( ('route', {'layers': route_layer}) )
      # conv
      cout = no * len(anchors[i])//2
      assert(cout == 255)
      conv_layer = convert_conv(cins[i], cout, 1,1,1, g=1, act='linear', bn=0)
      layers.append(conv_layer)
      # yolo
      yolo_config = {}
      mask_end = mask_start + len(anchors[i])//2
      yolo_config['mask'] = list(range(mask_start, mask_end))
      mask_start = mask_end
      yolo_config['anchors'] = total_anchors
      yolo_config['classes'] = nc
      yolo_config['num'] = 9
      yolo_config['jitter'] = 0.3
      yolo_config['ignore_thresh'] = 0.7
      yolo_config['truth_thresh'] = 1
      yolo_config['random'] = 1
#      yolo_config['scale_x_y'] = 1.05
#      yolo_config['iou_thresh'] = 0.213
#      yolo_config['cls_normalizer'] = 1.0
#      yolo_config['iou_normalizer'] = 0.07
#      yolo_config['iou_loss'] = 'ciou'
#      yolo_config['nms_kind'] = 'greedynms'
#      yolo_config['beta_nms'] = 0.6
      layers.append( ('yolo', yolo_config) )
    return layers

def write_head():
    layer_config = {}
    layer_config['batch'] = 1
    layer_config['subdivisions'] = 1
    layer_config['width'] = 640
    layer_config['height'] = 640
    layer_config['channels'] = 3
    layer_config['momentum'] = 0.9
    layer_config['decay'] = 0.0005
    layer_config['angle'] = 0
    layer_config['saturation'] = 1.5
    layer_config['exposure'] = 1.5
    layer_config['hue'] = .1
    
    layer_config['learning_rate'] = 0.001
    layer_config['burn_in'] = 1000
    layer_config['max_batches'] = 500200
    layer_config['policy'] = 'steps'
    layer_config['steps'] = [400000,450000]
    layer_config['scales'] = [.1,.1]
    return ('net', layer_config)


def parse_cfg(arch, save_file):
    # Read cfg file in type of YOLO-v5
    anchors, nc, gd, gw = arch['anchors'], arch['nc'], arch['depth_multiple'], arch['width_multiple']
#    gd = 0.33; gw = 0.75
    print(gd, gw)
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    new_arch, layer_map, ch = [], {}, [arch.get('ch', 3)]  # new_arch, layer_number_map
    for i, (f, n, m, args) in enumerate(arch['backbone'] + arch['head']):  # from, number, module, args
        for j, a in enumerate(args):
            try: args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except: pass
        if isinstance(args[-1], dict): args_dict = args[-1]; args = args[:-1]
        else: args_dict = {}

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in ['Conv', 'SepConv']:
            c2, k, d, s = args[0:4]
            if c2 != no:  # if not output
              if isinstance(c2, list): 
                c2 = [int(make_divisible(c * gw, 8)) for c in c2]
              else: c2 = make_divisible(c2*gw, 8)
            cin = ch[f]
            if m == 'Conv': g = 1
            elif m == 'SepConv': g = cin
            else: raise(ValueError("%s type is not Conv"%m))
            layers = [convert_conv(cin, c2, k, d, s, g)]
        elif m in ['C3']:
            c2, ks, ds = args[0:3]
            if c2 != no:  # if not output
              if isinstance(c2, list): 
                c2 = [int(make_divisible(c * gw, 8)) for c in c2]
              else: c2 = make_divisible(c2*gw, 8)
            shortcut = True if len(args)<4 else args[3]
            cin = ch[f]
            layers = convert_c3(n, cin, c2, ks, ds, shortcut, args_dict)
        elif m in ['Focus']:
            c1, c2, k = ch[f], args[0], args[1]
            if c2 != no:  # if not output
              if isinstance(c2, list): 
                c2 = [int(make_divisible(c * gw, 8)) for c in c2]
              else: c2 = make_divisible(c2*gw, 8)
            layers = convert_Focus(c1, c2, k)
        elif m in ['SPP']:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
              if isinstance(c2, list): 
                c2 = [int(make_divisible(c * gw, 8)) for c in c2]
              else: c2 = make_divisible(c2*gw, 8)
            layers = convert_SPP(c1, c2, ks=(5,9,13))
        elif m in ['FF']:
            c1s = [ch[x] for x in f]
            route_layers = [layer_map[x] if x>0 else layer_map[x+i] for x in f]
            c2, strides, ks, ds = args[0:4]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            layers = convert_FF(c1s, c2, route_layers, strides, ks, ds, args_dict)
        elif m in ['Detect']:
            cins = [ch[x] for x in f]
            route_layers = [layer_map[x] if x>0 else layer_map[x+i] for x in f]
            anchors = [list(range(args[1] * 2))] * len(f) if isinstance(args[1], int) else args[1]
            layers = convert_Detect(route_layers, nc, anchors, cins)

        new_arch.extend(layers)
        if i == 0:
            ch = []
        if isinstance(c2, list): ch.append(c2[-1])
        else: ch.append(c2)
        layer_map[i] = len(new_arch)-1

    # save new arch
    new_arch.insert(0, write_head())

    with open(save_file, 'w') as f:
      for layer in new_arch:
        f.write('[%s]\n'%layer[0])
        for k, v in layer[1].items():
          string = '%s='%k
          if isinstance(v, list): str_v = str(v).strip('[').strip(']')
          else: str_v = str(v)
          string = '%s%s\n'%(string, str_v)
          f.write(string)
        f.write('\n')
    return layer_map

def pre_process(state_dict):
    for idx, (name, v) in enumerate(state_dict.items()):
      if idx == 0:
        print("Process the weight of Focus in %s"%name)
        cout, cin, k,_ = v.shape
        cin_new = cin // 4
        v_new = torch.zeros_like(v).view(cout, cin_new, k*2, k*2)
        v_new[:,:,0:2*k:2, 0:2*k:2] = v.data[:,:cin_new,:,:]
        v_new[:,:,1:2*k:2, 0:2*k:2] = v.data[:,cin_new:2*cin_new,:,:]
        v_new[:,:,0:2*k:2, 1:2*k:2] = v.data[:,2*cin_new:3*cin_new,:,:]
        v_new[:,:,1:2*k:2, 1:2*k:2] = v.data[:,3*cin_new:,:,:]
        v.data = v_new

def parse_weight(model, path):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346        
#        self.version.tofile(f)  # (int32) version info: major, minor, revision
#        self.seen.tofile(f)  # (int64) number of images seen during training
         np.array([0, 2, 5], dtype=np.int32).tofile(f)
         np.array([0], dtype=np.int64).tofile(f)

         cnt = 0
         params = []
         for k, v in model.named_modules():
             if isinstance(v, nn.Conv2d):
                 if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                   v.bias.data.cpu().numpy().tofile(f)
                   v.weight.data.cpu().numpy().tofile(f)
                   cnt += 1
                 else:
                   assert(len(params)<=1) # only for dw,gw-conv, params have item
                   cnt += len(params)
                   for p in params: p.tofile(f)
                   params=[]
                   weight = v.weight.data
                   params.append(weight.cpu().numpy())
             elif isinstance(v, nn.BatchNorm2d):
                 v.bias.data.cpu().numpy().tofile(f)
                 v.weight.data.cpu().numpy().tofile(f)
                 v.running_mean.data.cpu().numpy().tofile(f)
                 v.running_var.data.cpu().numpy().tofile(f)
                 for p in params: p.tofile(f)
                 params = []
                 cnt += 1
         assert(len(params) == 0)
    print("Save %d params"%cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='cfg file in type of YOLO-v5')
    parser.add_argument('--weight_file', type=str, default=None, help='pt file')
    parser.add_argument('--save_dir', type=str, default='models/YOLO_cfg', help='save dir for cfg file in type of YOLO-v4')
    args = parser.parse_args()
#    file_name = args.cfg_file.split('/')[-1]
    file_name = args.cfg_file.split('/')[-4]

    cfg_save_file = os.path.join(args.save_dir, '%s.cfg'%file_name)
    # convert cfg
    with open(args.cfg_file) as f:
        arch = yaml.load(f, Loader=PrettySafeLoader)  # model dict
    layer_map = parse_cfg(arch, cfg_save_file)

    weight = torch.load(args.weight_file)['model']
    # conver weight to YOLOv4 pt style
    # since in C3, the ordering of cv2, cv3 and m differs from new_cfg_file, so we reproduce state_dict
    model = Model(args.cfg_file, ch=3, nc=80, anchors=None)  # create
    exclude = ['total_ops', 'total_params'] # exclude keys
    state_dict = weight.float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=True)  # load

    pt_save_file = os.path.join(args.save_dir, '%s.pt'%file_name)
    torch.save(model.state_dict(), pt_save_file)

#    # eval model
#    args.single_cls = False
#    data = './data/coco.yaml'
#    with open(data) as f:
#        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
#    path = data['val']  
#    dataloader = create_dataloader(path, 640, 32, 32, args, pad=0.5, rect=True,
#                                       prefix=colorstr('val: '))[0]
#    results, maps, times = test.test('./data/coco.yaml',
#                                     batch_size=32,
#                                     imgsz=640,
#                                     model=model,
##                                     save_dir = Path(''))
#                                     dataloader=dataloader
#                                    )
#    print(results)
    '''
    # debug
    sd = model.state_dict()
    bn_weight = []
    for idx, (k,v) in enumerate(sd.items()):
      print(idx, k, v.view(-1)[-10:])
      bn_weight.append(v)
      if idx > 5: break
    x = torch.ones((1, model.yaml.get('ch', 3), 640, 640), device=next(model.parameters()).device)  # input
    model.eval()
    y = []  # outputs
    for m in model.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x)  # save output
#    print(y[20])
#    print(y[20].shape)
#    print(layer_map[20])
    print(y[-1][0])
    print(y[-1][0].shape)
    assert 0
    '''

#    state_dict = model.state_dict()
#    exclude = ['anchor', 'total_ops', 'total_params'] # exclude keys
#    state_dict = intersect_dicts(state_dict, state_dict, exclude=exclude)  # intersect
#    # convert weight
#    w_save_file = os.path.join(args.save_dir, '%s.weights'%file_name)
#    parse_weight(model, w_save_file)

