import argparse
import logging
import sys
from copy import deepcopy
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from models import darts_cell
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(Conv_search_merge(x, self.no * self.na, kd=[(1,1)], candidate_e=[1.], s=1, bias=True) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5_search.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
#            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            self.cfg = cfg
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self._arch_parameters, self.op_arch_parameters, self.ch_arch_parameters, self.edge_arch_parameters, self.search_space_per_layer = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # check the number of architecture parameters
        cnt = 0
        for n, b in self.model.named_buffers():
          if 'alpha' in n: cnt+=1
        assert (cnt==len(self._arch_parameters))
        cnt = 0
        for alpha in self.op_arch_parameters:
           if alpha is not None: cnt+=1
        for alpha in self.ch_arch_parameters:
           if alpha is not None: cnt+=1
        for alpha in self.edge_arch_parameters:
           if alpha is not None: cnt+=1
        assert (cnt==len(self._arch_parameters))
        assert(len(self.search_space_per_layer) == len(self.op_arch_parameters))
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def update_arch_parameters(self):
        state_dict = self.state_dict()
        self._arch_parameters = []
        for key in state_dict.keys():
          if 'alpha' in key: self._arch_parameters.append(state_dict[key])
        return self._arch_parameters

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def genotype(self):
        op_geno=[]; op_geno_idx = []
        ch_geno = []; ch_geno_idx = []
        edge_geno = []; edge_geno_idx = []
        for alphas in self.op_arch_parameters:
            if alphas.dim() == 1:
              op_geno_idx.append(alphas.argmax(dim=-1).item())
            else:
              op_geno_idx.append(alphas)
        for alphas in self.ch_arch_parameters:
            if alphas is None: ch_geno_idx.append(None)
            else: ch_geno_idx.append(alphas.argmax(dim=-1).item())
        for alphas in self.edge_arch_parameters:
            edge_geno_idx.append([x.item() for x in torch.topk(alphas, k=2, dim=-1)[1]])
        # new yaml
        with open(self.cfg) as f:
            model_yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        idx_op = 0
        idx_ch = 0
        idx_edge = 0
        gd = model_yaml['depth_multiple']
        for i, tmp in enumerate(model_yaml['backbone'] + model_yaml['head']):
            if isinstance(tmp[3][-1], dict): # del unused variables
               for key in ['gumbel_channel']:
                 if key in tmp[3][-1].keys(): del tmp[3][-1][key]
            if tmp[2] in ['Conv_search', 'Bottleneck_search', 'Conv_search_merge', 'Bottleneck_search_merge', 'SepConv_search_merge']:
               n = tmp[1]
               n = max(round(n * gd), 1) if n > 1 else n  # depth gain
               func_p = tmp[3]
               Cout = func_p[0]
               tmp[2] = tmp[2].split('_')[0] # set name
               # set kernel-size and dilation-ratio and channel
               k = []; d = []; e = [];
               for j in range(n):
                 k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                 d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                 if ch_geno_idx[idx_ch+j] is None: e.append(1.0) # original YOLOv5 uses e=1.0
                 else: e.append(func_p[2][ch_geno_idx[idx_ch+j]])
               op_geno.append(list(zip(k,d)))
               ch_geno.append(e)
               if n == 1: k = k[0]; d=d[0]; e=e[0];
               tmp[3][1] = k
#               tmp[3].insert(2, d)
               tmp[3][2] = d # originally, tmp[3][2] is candidate_e, which is useless for full-train
               if tmp[2] in ['Bottlenect']:
                 if isinstance(tmp[3][-1], dict): tmp[3][-1]['e_bottleneck'] = e
                 else: tmp[3].append({'e_bottleneck':e})
               else: tmp[3][0] = Cout * e 
               idx_op += n; idx_ch += n
            elif tmp[2] in ['C3_search', 'C3_search_merge']:
               n = tmp[1]
               n = max(round(n * gd), 1) if n > 1 else n  # depth gain
               func_p = tmp[3]
               Cout = func_p[0]
               candidate_e = func_p[2]
               tmp[2] = tmp[2].split('_')[0] # set name
               # set kernel-size and dilation-ratio and channel
               k = []; d = []; e = [];
               for j in range(n):
                 k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                 d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                 if ch_geno_idx[idx_ch+j] is None: e.append(1.0) # original YOLOv5 uses e=1.0
                 else: e.append(candidate_e[ch_geno_idx[idx_ch+j]])
               op_geno.append(list(zip(k,d)))
               ch_geno.append(deepcopy(e))
               if n == 1: k = k[0]; d=d[0]; e=e[0];
               tmp[3][1] = k
#               tmp[3].insert(2, d)
               tmp[3][2] = d
               if isinstance(tmp[3][-1], dict): tmp[3][-1]['e_bottleneck'] = e
               else: tmp[3].append({'e_bottleneck':e})
               # for c2
               if isinstance(func_p[-1], dict) and func_p[-1].get('search_c2', False):
                 if isinstance(func_p[-1]['search_c2'], list):
                   tmp[3][0] = Cout * func_p[-1]['search_c2'][ch_geno_idx[idx_ch+n]]
                   ch_geno[-1].append(func_p[-1]['search_c2'][ch_geno_idx[idx_ch+n]])
                 else:
                   tmp[3][0] = Cout * candidate_e[ch_geno_idx[idx_ch+n]]
                   ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+n]])
                 del tmp[3][-1]['search_c2']
                 idx_ch += n+1
               else: idx_ch += n
               idx_op += n
            elif tmp[2] == 'AFF':
               tmp[2] = 'FF'
               all_edges = tmp[0]
               Cout, all_strides, all_kds = tmp[3][0:3]
               if isinstance(tmp[3][-1], dict):
                  candidate_e = tmp[3][-1].get('candidate_e', None)
                  separable = tmp[3][-1].get('separable', False)
               else: candidate_e = None; separable = False
               edges = []; ks= []; ds = []; strides = [];
               for j, idx in enumerate(edge_geno_idx[idx_edge]):
                  edges.append(all_edges[idx])
                  strides.append(all_strides[idx])
                  ks.append(all_kds[op_geno_idx[idx_op+idx]][0])
                  ds.append(all_kds[op_geno_idx[idx_op+idx]][1])
               edge_geno.append(edges)
               op_geno.append(list(zip(ks, ds)))
               ch_geno.append([1.0 for _ in range(len(edges))])
               # for Cout
               if ch_geno_idx[idx_ch+len(all_edges)] is not None:
                   Cout = Cout * candidate_e[ch_geno_idx[idx_ch+len(all_edges)]]
                   ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+len(all_edges)]])
               args_dict = {'separable': separable}
               tmp[3] = [Cout, strides, ks, ds, args_dict]
               tmp[0] = edges
               idx_op += len(all_edges)
               idx_ch += len(all_edges)+1
               idx_edge += 1
            elif tmp[2] == 'SPP_search':
               tmp[2] = tmp[2].split('_')[0] # set name
            elif tmp[2] in ['Cells_search', 'Cells_search_merge']:
               tmp[2] = tmp[2].split('_')[0] # set name
               steps, multiplier, C, reduction, reduction_prev = tmp[3][0:5]
               op_alpha = op_geno_idx[idx_op]
               genotype, concat = darts_cell.genotype(op_alpha, steps, multiplier, num_input=len(tmp[0]))
               tmp[3] = [genotype, concat, C, reduction, reduction_prev]
               op_geno.append([genotype, concat])
               ch_geno.append([1])
               idx_op += 1
               idx_ch += 1
        assert(idx_ch == len(ch_geno_idx))
        assert(idx_op == len(op_geno_idx))
        assert(idx_edge == len(edge_geno_idx))
        geno = [op_geno, ch_geno, edge_geno] # split the alpha_op and alpha_channal
        model_yaml['geno'] = geno
        return geno, model_yaml

    def display_alphas(self):
        op_alphas = []
        channel_alphas = []
        edge_alphas = []
#        for i, alphas in enumerate(self._arch_parameters):
#            if i % 2 == 0: op_alphas.append(torch.nn.functional.softmax(alphas, dim=-1))
#            else: channel_alphas.append(torch.nn.functional.softmax(alphas, dim=-1))
        for i, alphas in enumerate(self.op_arch_parameters):
            op_alphas.append(torch.nn.functional.softmax(alphas, dim=-1))
        for i, alphas in enumerate(self.ch_arch_parameters):
            if alphas is None: channel_alphas.append(None)
            else: channel_alphas.append(torch.nn.functional.softmax(alphas, dim=-1))
        for i, alphas in enumerate(self.edge_arch_parameters):
            edge_alphas.append(torch.nn.functional.softmax(alphas, dim=-1))
        print("op alphas")
        for a in op_alphas:
            print(a)
        print("channel alphas")
        for a in channel_alphas:
            print(a)
        if len(edge_alphas)>0:
          print("edge alphas")
          for a in edge_alphas:
              print(a)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    arch_parameters = []
    op_arch_parameters = []
    ch_arch_parameters = []
    edge_arch_parameters = []
    search_space_per_layer = [] # for each layer that need to be searched, we construct a dict to record its candidate kernel sizes, dilation ratios and channel ratios
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        if isinstance(args[-1], dict): args_dict = args[-1]; args = args[:-1]
        else: args_dict = {}

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, Bottleneck_search, GhostBottleneck, SPP, DWConv, MixConv2d, Conv_search, Focus, CrossConv, BottleneckCSP, C3, C3_search, Conv_search_merge, Bottleneck_search_merge, C3_search_merge, SPP_search]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3_search, C3_search_merge]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in [Cells_search, Cells_search_merge, Cells]:
            if len(f) == 2:
              c_prev_prev, c_prev, c2 = ch[f[-2]], ch[f[-1]], args[2]
            else:
              c_prev_prev, c_prev, c2 = None, ch[f[-1]], args[2]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args[2] = c2
            args.insert(2, c_prev_prev)
            args.insert(3, c_prev)
            args_dict['N'] = n
            n = 1
        elif m in [AFF, FF]:
            c1s = [ch[x] for x in f]
            c2 = args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1s, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args, **args_dict) for _ in range(n)]) if n > 1 else m(*args, **args_dict)  # module
        if m in [Conv_search, Bottleneck_search, C3_search, Conv_search_merge, Bottleneck_search_merge, C3_search_merge, AFF, Cells_search, Cells_search_merge]:
          m_list = [m_] if n==1 else m_
          for tmp in m_list: 
            arch_parameters.extend(tmp.get_alphas())
            op_arch_parameters.extend(tmp.get_op_alphas())
            ch_arch_parameters.extend(tmp.get_ch_alphas())
            d = {'kd': args[1], 'ch_ratio': args[2]}
            search_space_per_layer.extend([d for _ in range(len(tmp.get_op_alphas()))])
        if m in [AFF]:
          edge_arch_parameters.extend(m_.get_edge_alphas())

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s%-30s' % (i, f, n, np, t, args, args_dict))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if m in [Conv_search_merge]:
          ch.append(int(c2*max(args[3])))
        elif m in [C3_search, Bottleneck_search, C3_search_merge, Bottleneck_search_merge]:
          ch.append(c2)
        elif m in [Cells_search, Cells_search_merge]:
          ch.append(c2*args[1])
        else: ch.append(c2)
    return nn.Sequential(*layers), sorted(save), arch_parameters, op_arch_parameters, ch_arch_parameters, edge_arch_parameters, search_space_per_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard


