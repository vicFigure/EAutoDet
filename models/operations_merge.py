import torch
import torch.nn as nn
from torch.autograd import Variable

PRIMITIVES_merge = [
#    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'merge_sep_5x5',
]

OPS = {
  'none' : lambda C, stride, affine, single_sepConv, with_d5: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, single_sepConv, with_d5: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine, single_sepConv, with_d5: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine, single_sepConv, with_d5: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'merge_sep_5x5' : lambda C, stride, affine, single_sepConv, with_d5: MergeSepConv(C, C, 5, stride, 4, affine=affine, single=single_sepConv, with_d5=with_d5),
  'sep_conv_3x3' : lambda C, stride, affine, single_sepConv, with_d5: SepConv(C, C, 3, stride, 1, affine=affine, single=single_sepConv),
  'sep_conv_5x5' : lambda C, stride, affine, single_sepConv, with_d5: SepConv(C, C, 5, stride, 2, affine=affine, single=single_sepConv),
  'sep_conv_7x7' : lambda C, stride, affine, single_sepConv, with_d5: SepConv(C, C, 7, stride, 3, affine=affine, single=single_sepConv),
  'dil_conv_3x3' : lambda C, stride, affine, single_sepConv, with_d5: DilConv(C, C, 3, stride, 2, 2, affine=affine, single=single_sepConv),
  'dil_conv_5x5' : lambda C, stride, affine, single_sepConv, with_d5: DilConv(C, C, 5, stride, 4, 2, affine=affine, single=single_sepConv),
  'conv_3x3' : lambda C, stride, affine, single_sepConv, with_d5: ReLUConvBN(C, C, 3, stride, 1, affine=affine, dilation=1),
  'conv_5x5' : lambda C, stride, affine, single_sepConv, with_d5: ReLUConvBN(C, C, 5, stride, 2, affine=affine, dilation=1),
  'conv_3x3_dil' : lambda C, stride, affine, single_sepConv, with_d5: ReLUConvBN(C, C, 3, stride, 2, affine=affine, dilation=2),
  'conv_5x5_dil' : lambda C, stride, affine, single_sepConv, with_d5: ReLUConvBN(C, C, 5, stride, 4, affine=affine, dilation=2),
  'conv_7x1_1x7' : lambda C, stride, affine, single_sepConv, with_d5: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, dilation=1):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.SiLU(),
#      nn.LeakyReLU(0.0, inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    x = torch.nn.functional.relu(x, inplace=False)
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, single=False):
    super(DilConv, self).__init__()
    if single:
      self.op = nn.Sequential(
#        nn.ReLU(inplace=False),
#        nn.SiLU(),
        nn.LeakyReLU(0.0, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        )
    else:
      self.op = nn.Sequential(
#        nn.ReLU(inplace=False),
#        nn.SiLU(),
        nn.LeakyReLU(0.0, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_in, affine=affine),
  
        # it's different from original darts
#        nn.ReLU(inplace=False),
#        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=C_in, bias=False),
#        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#        nn.BatchNorm2d(C_out, affine=affine),
        )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, single=False):
    super(SepConv, self).__init__()
    if single:
      self.op = nn.Sequential(
#        nn.ReLU(inplace=False),
#        nn.SiLU(),
        nn.LeakyReLU(0.0, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        )
    
    else:
      self.op = nn.Sequential(
#        nn.ReLU(inplace=False),
#        nn.SiLU(),
        nn.LeakyReLU(0.0, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_in, affine=affine),
#        nn.ReLU(inplace=False),
#        nn.SiLU(),
        nn.LeakyReLU(0.0, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        )

  def forward(self, x):
    return self.op(x)

class MergeSepConv_v0(nn.Module):
  '''
  First, use a normal conv weights to replace the separable conv
  Second, computes the weighted average of the 4 normal conv weights together
  Third, conduct the single convolution
  '''
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(MergeSepConv_v0, self).__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.stride = stride
    self.padding = padding
    self.affine = affine

    self.relu1 = nn.ReLU(inplace=False)
    self._init_weights()
    self.bn1 = nn.BatchNorm2d(C_in, affine=affine)

  def _init_weights(self):
    C_in = self.C_in
    C_out = self.C_out
    self.depth_w3 = nn.Parameter(torch.Tensor(C_in, 1, 3, 3))
    self.point_w3 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.depth_w5 = nn.Parameter(torch.Tensor(C_in, 1, 5, 5))
    self.point_w5 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.depth_dw3 = nn.Parameter(torch.Tensor(C_in, 1, 3, 3))
    self.point_dw3 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.depth_dw5 = nn.Parameter(torch.Tensor(C_in, 1, 5, 5))
    self.point_dw5 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    
    torch.nn.init.kaiming_normal_(self.depth_w3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_w3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.depth_w5,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_w5,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.depth_dw3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_dw3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.depth_dw5,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_dw5,  mode='fan_in')

  def forward(self, x, weight):
    x = self.relu1(x)
    padding = 4 
    merge_kernel_1 = self.get_merge_kernel(weight)
    x = torch.nn.functional.conv2d(x, merge_kernel_1, stride=self.stride, padding=padding, dilation=1, groups=1)
    x = self.bn1(x)
    return x

  def merge_depth_point(self, depth, point):
    d_tmp = depth.permute(1,0,2,3)
    w_merge = point * d_tmp
    return w_merge

  def get_merge_kernel(self, alphas):
    # merge the depth-wise conv and point-wise conv
    merge_w3 = self.merge_depth_point(self.depth_w3, self.point_w3)
    merge_w5 = self.merge_depth_point(self.depth_w5, self.point_w5)
    merge_dw3 = self.merge_depth_point(self.depth_dw3, self.point_dw3)
    merge_dw5 = self.merge_depth_point(self.depth_dw5, self.point_dw5)

    # make all the conv kernel to 9*9
    w3_pad = torch.nn.functional.pad(merge_w3, (3,3,3,3), "constant", value=0)
    w5_pad = torch.nn.functional.pad(merge_w5, (2,2,2,2), "constant", value=0)
    dw3 = torch.zeros_like(w5_pad)
    dw3[:,:,2:7:2,2:7:2] = merge_dw3
    dw5 = torch.zeros_like(w5_pad)
    dw5[:,:,0:9:2,0:9:2] = merge_dw5

    # merge all the conv kernels
    alpha3, alpha5, alphad3, alphad5 = alphas
    merge_kernel = w5_pad*alpha5 + w3_pad*alpha3 + dw3*alphad3 + dw5*alphad5
    return merge_kernel

class MergeSepConv_v1(nn.Module):
  '''
  only share depth-wise conv
  First, compute the weighted average of point-wise weights according to Mask
  Second, compute the Hadamad Mul of shared depth-wise weight and the merged point-wise weights
  Third, conduct the single convolution
  '''
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(MergeSepConv_v1, self).__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.stride = stride
    self.padding = padding
    self.affine = affine

    self.relu1 = nn.ReLU(inplace=False)
    self._init_weights()
    self.bn1 = nn.BatchNorm2d(C_in, affine=affine)

  def _init_weights(self):
    C_in = self.C_in
    C_out = self.C_out
    self.depth_w = nn.Parameter(torch.Tensor(C_in, 1, 9, 9))
    self.point_w3 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.point_w5 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.point_dw3 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
    self.point_dw5 = nn.Parameter(torch.Tensor(C_in, C_out, 1, 1))
   
    torch.nn.init.kaiming_normal_(self.depth_w,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_w3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_w5,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_dw3,  mode='fan_in')
    torch.nn.init.kaiming_normal_(self.point_dw5,  mode='fan_in')

  def forward(self, x, weight):
    x = self.relu1(x)
    padding = 4 
    merge_kernel_1 = self.get_merge_kernel(weight)
    x = torch.nn.functional.conv2d(x, merge_kernel_1, stride=self.stride, padding=padding, dilation=1, groups=1)
    x = self.bn1(x)
    return x

  def merge_point(self, alphas):
    alpha3, alpha5, alphad3, alphad5 = alphas
    dtype = self.depth_w.dtype
    device = self.depth_w.device
    mask_w3 = torch.zeros((1,1,9,9), dtype=depth, device=device)
    mask_w3[:,:,3:6,3:6] = 1.
    mask_w5 = torch.zeros((1,1,9,9), dtype=depth, device=device)
    mask_w5[:,:,2:7,2:7] = 1.
    mask_dw3 = torch.zeros((1,1,9,9), dtype=depth, device=device)
    mask_dw3[:,:,2:7:2,2:7:2] = 1.
    mask_dw5 = torch.zeros((1,1,9,9), dtype=depth, device=device)
    mask_dw5[:,:,0:9:2,0:9:2] = 1.

    p_merge = alpha3*mask_w3*self.point_w3 + alpha5*mask_w5*self.point_w5 + alphad3*mask_dw3*self.point_dw3 + alphad5*mask_dw5*mask_dw5
    return p_merge

  def get_merge_kernel(self, alphas):
    # merge the depth-wise conv and point-wise conv
    merge_point = self.merge_point(alphas)

    # merge all the conv kernels
    d_tmp = self.depth_w.permute(1,0,2,3)
    merge_kernel = merge_point * d_tmp
    return merge_kernel


class MergeSepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, single=False, with_d5=True):
    super(MergeSepConv, self).__init__()
    self.stride = stride
    self.padding = padding
    self.affine = affine
    self.single = single
    self.with_d5 = with_d5
    if self.with_d5:
      self.kernel_size = kernel_size*2-1
    else:
      self.kernel_size = kernel_size

#    self.act1 = nn.ReLU(inplace=False)
#    self.act1 = nn.SiLU()
#    self.act1 = nn.LeakyReLU(0.0, inplace=False)
#    self.depth_conv1 = nn.Conv2d(C_in, C_in, kernel_size=self.kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
    tmp1 = torch.Tensor(C_in, 1, self.kernel_size, self.kernel_size)
    torch.nn.init.kaiming_normal_(tmp1,  mode='fan_in')
    self.depth_weight1 = nn.Parameter(tmp1)
    C_tmp = C_out if single else C_in
    self.point_conv1 = nn.Conv2d(C_in, C_tmp, kernel_size=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(C_tmp, affine=affine)

    '''
    self.Mask5_tmp = torch.ones(self.kernel_size, requires_grad=True)
    self.Mask3_tmp = torch.zeros(self.kernel_size, requires_grad=True)
    self.Maskd3_tmp = torch.zeros(self.kernel_size, requires_grad=True)
    self.Mask3[1:4,1:4] = 1.
    self.Maskd3[0:5:2,0:5:2] = 1.
    self.Mask5_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.Mask3_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.Maskd3_tmp = self.Mask5_tmp.unsqueeze(0).unsqueeze(0)
    self.register_buffer("Mask5", self.Mask5_tmp)
    self.register_buffer("Mask3", self.Mask3_tmp)
    self.register_buffer("Maskd3", self.Maskd3_tmp)
    if self.with_d5:
      self.Maskd5_tmp = torch.ones(self.kernel_size)
    '''

    if not single:
#      self.act2 = nn.ReLU(inplace=False)
#      self.act2 = nn.SiLU()
#      self.act2 = nn.LeakyReLU(0.0, inplace=False)
#      self.depth_conv2 = nn.Conv2d(C_in, C_in, kernel_size=self.kernel_size, stride=1, padding=padding, groups=C_in, bias=False)
      tmp2 = torch.Tensor(C_in, 1, self.kernel_size, self.kernel_size)
      torch.nn.init.kaiming_normal_(tmp2,  mode='fan_in')
      self.depth_weight2 = nn.Parameter(tmp2)
      self.point_conv2 = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
      self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, weight):
#    x = self.act1(x)
    x = torch.nn.functional.relu(x)
    padding = 4 if self.with_d5 else 2
    C_in = x.size(1)
    w5_1 = self.depth_weight1
    merge_kernel_1 = self.get_merge_kernel(w5_1, weight, with_d5=self.with_d5)
    x = torch.nn.functional.conv2d(x, merge_kernel_1, stride=self.stride, padding=padding, dilation=1, groups=C_in)
    x = self.point_conv1(x)
    x = self.bn1(x)

    if not self.single:
#      x = self.act2(x)
      x = torch.nn.functional.relu(x)
      w5_2 = self.depth_weight2
      merge_kernel_2 = self.get_merge_kernel(w5_2, weight, with_d5=self.with_d5)
      x = torch.nn.functional.conv2d(x, merge_kernel_2, stride=1, padding=padding, dilation=1, groups=C_in)
      x = self.point_conv2(x)
      x = self.bn2(x)
    
    return x

  def get_merge_kernel(self, w5, alphas, with_d5=True):
    if with_d5:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3, alphad5 = alphas
        w5_pad = w5[:,:,2:7,2:7]
        w5_pad = torch.nn.functional.pad(w5_pad, (2,2,2,2), "constant", value=0)
    
        w3 = w5[:,:,3:6,3:6]
        w3_pad = torch.nn.functional.pad(w3, (3,3,3,3), "constant", value=0)
    
#        dw3 = Variable(torch.zeros(Cout, C, 5, 5)).cuda()
        dw3 = torch.zeros_like(w5)
        dw3[:,:,2:7:2,2:7:2] = w5[:,:,2:7:2,2:7:2]
    
#        dw5 = Variable(torch.zeros(Cout, C, 9, 9)).cuda()
#        dw5 = torch.zeros_like(w5, requires_grad=True)
        dw5 = torch.zeros_like(w5)
        dw5[:,:,0:9:2,0:9:2] = w5[:,:,0:9:2,0:9:2]
        merge_kernel = w5_pad*alpha5 + w3_pad*alpha3 + dw3*alphad3 + dw5*alphad5
    else:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3 = alphas
    
        w3 = w5[:,:,1:4,1:4]
        w3_pad = torch.nn.functional.pad(w3, (1,1,1,1), "constant", value=0)
    
#        dw3 = Variable(torch.zeros(Cout, C, 5, 5)).cuda()
        dw3 = torch.zeros_like(w5)
        dw3[:,:,0:5:2,0:5:2] = w5[:,:,0:5:2,0:5:2]
    
        merge_kernel = w5*alpha5 + w3_pad*alpha3 + dw3*alphad3
    return merge_kernel
    
  def get_merge_kernel_mask(self, w5, alphas, with_d5=True):
    if with_d5:
        raise(ValueError("for mask forward, with_d5 should be False"))
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3, alphad5 = alphas
        w5_pad = torch.nn.functional.pad(w5, (2,2,2,2), "constant", value=0)
        merge_kernel = w5_pad*(alpha5*self.Mask5 + alpha3*self.Mask3 + alphad3*self.Maskd3)
    else:
        Cout,C,_,_ = w5.shape
        alpha3, alpha5, alphad3 = alphas

        self.Mask5_tmp = torch.ones_like(w5, requires_grad=True)
        self.Mask3_tmp = torch.zeros_like(w5, requires_grad=True)
        self.Maskd3_tmp = torch.zeros_like(w5, requires_grad=True)
        self.Mask3[1:4,1:4] = 1.
        self.Maskd3[0:5:2,0:5:2] = 1.
    
        merge_kernel = w5*(alpha5*self.Mask5 + alpha3*self.Mask3 + alphad3*self.Maskd3)
    return merge_kernel

class MergeConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(MergeConv, self).__init__()
    self.stride = stride
    self.padding = padding
    self.affine = affine

    self.relu1 = nn.ReLU(inplace=False)
    tmp1 = torch.Tensor(C_in, C_out, 9, 9)
    torch.nn.init.kaiming_normal_(tmp1,  mode='fan_in')
    self.weight = nn.Parameter(tmp1)
    self.bn1 = nn.BatchNorm2d(C_out, affine=affine)


  def forward(self, x, weight):
    x = self.relu1(x)
    padding = 4
    merge_kernel_1 = self.get_merge_kernel(self.weight, weight)
    x = torch.nn.functional.conv2d(x, merge_kernel_1, stride=self.stride, padding=padding, dilation=1, groups=1)
    x = self.bn1(x)

    return x

  def get_merge_kernel(self, w_base, alphas):
    Cout,C,_,_ = w_base.shape
    alpha3, alpha5, alphad3, alphad5 = alphas
    w3 = torch.zeros_like(w_base)
    w3[:,:,3:6,3:6] = w_base[:,:,3:6,3:6]
    w5 = torch.zeros_like(w_base)
    w5[:,:,2:7,2:7] = w_base[:,:,2:7,2:7]
    dw3 = torch.zeros_like(w_base)
    dw3[:,:,2:7:2,2:7:2] = w_base[:,:,2:7:2,2:7:2]
    dw5 = torch.zeros_like(w_base)
    dw5[:,:,0:9:2,0:9:2] = w_base[:,:,0:9:2,0:9:2]
    merge_kernel = w5*alpha5 + w3*alpha3 + dw3*alphad3 + dw5*alphad5
    
    return merge_kernel
    
class MergeAllSep(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(MergeAllSep, self).__init__()
    self.stride = stride
    self.padding = padding
    self.affine = affine

    self.relu = nn.ReLU(inplace=False)
    tmp1 = torch.Tensor(C_in, 1, 9, 9)
    torch.nn.init.kaiming_normal_(tmp1,  mode='fan_in')
    self.depth_weight = nn.Parameter(tmp1)
    tmp2= torch.Tensor(C_out, C_in, 1, 1)
    torch.nn.init.kaiming_normal_(tmp2,  mode='fan_in')
    self.point_weight = nn.Parameter(tmp2)
    self.max_pool = nn.MaxPool2d(3, stride=stride, padding=1)
    self.avg_pool = nn.AvgPool2d(3, stride=stride, padding=1)
    if stride == 2:
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn_skip = nn.BatchNorm2d(C_out, affine=affine)
        
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.bn_avg = nn.BatchNorm2d(C_out, affine=affine)
    self.bn_max = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, weight):
    a_none, a_max, a_avg, a_skip, a_3, a_5, a_d3, a_d5 = weight
    a_max = weight[1]
    x = self.relu(x)
    C_in = x.size(1)
    merge_kernel = self.get_merge_kernel(weight)
    yconv = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=4, dilation=1, groups=1)
    ymax = a_max*self.max_pool(x)
    yavg = a_avg*self.avg_pool(x)
    ynone = a_none*0
#    y = y1 + y2
    yconv = self.bn(yconv)
    ymax = self.bn_max(ymax)
    yavg = self.bn_avg(yavg)
    if self.stride == 2:
#        yskip = a_skip * torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        yskip = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        yskip = a_skip * self.bn_skip(yskip)
    else:
        yskip = a_skip*x
    y = yconv + ymax + yavg + ynone + yskip
#    y = self.bn(y)
    return y

  def get_merge_kernel(self, alphas):
    cout, cin = self.point_weight.shape[0:2]
    dtype = self.depth_weight.dtype
    device = self.depth_weight.device
    if cout != cin:
        raise(ValueError("cout != cin, so skip-connect cannot be merged"))
    a_none, a_max, a_avg, a_skip, a_3, a_5, a_d3, a_d5 = alphas
    # merge the sep convs
    d_tmp = self.depth_weight.permute(1,0,2,3).contiguous()
    w5 = self.point_weight * d_tmp
    # conv
    w5_pad = w5[:,:,2:7,2:7]
    w5_pad = torch.nn.functional.pad(w5_pad, (2,2,2,2), "constant", value=0)
    
    w3 = w5[:,:,3:6,3:6]
    w3_pad = torch.nn.functional.pad(w3, (3,3,3,3), "constant", value=0)

    dw3 = torch.zeros_like(w5)
    dw3[:,:,2:7:2,2:7:2] = w5[:,:,2:7:2,2:7:2]
    
    dw5 = torch.zeros_like(w5)
    dw5[:,:,0:9:2,0:9:2] = w5[:,:,0:9:2,0:9:2]
    merge_kernel = w5_pad*a_5 + w3_pad*a_3 + dw3*a_d3 + dw5*a_d5
    return merge_kernel
    assert(0)
    if cout == cin:
        # avg
        wavg = torch.eye(cout, dtype=dtype, device=device).view(cout,cout,1)/9.
        wavg = wavg.repeat(1,1,9)
        wavg = wavg.view(cout,cout,3,3)
        wavg = torch.nn.functional.pad(wavg, (3,3,3,3), "constant", value=0)
#        # skip-connect
#        wskip = torch.eye(cout, dtype=dtype, device=device).view(cout,cout,1,1)
#        wskip = torch.nn.functional.pad(wskip, (4,4,4,4), "constant", value=0)
    
        merge_kernel = w5_pad*a_5 + w3_pad*a_3 + dw3*a_d3 + dw5*a_d5 + wavg*a_avg + 0*a_none
    else:
        raise(ValueError("cout != cin, so skip-connect cannot be merged"))
        merge_kernel = w5_pad*a_5 + w3_pad*a_3 + dw3*a_d3 + dw5*a_d5 + 0*a_none
    return merge_kernel

  def get_merge_kernel_mask(self, alphas):
    cout, cin = self.point_weight.shape[0:2]
    dtype = self.depth_weight.dtype
    device = self.depth_weight.device
    if cout != cin:
        raise(ValueError("cout != cin, so skip-connect cannot be merged"))
    a_none, a_max, a_avg, a_skip, a_3, a_5, a_d3, a_d5 = alphas
    # merge the sep convs
    d_tmp = self.depth_weight.permute(1,0,2,3).contiguous()
    w5 = self.point_weight * d_tmp
    # conv
    mask_5 = torch.zeros((1,1,9,9), dtype=dtype, device=device)
    mask_5[:,:,2:7,2:7] = 1.
    mask_3 = torch.zeros((1,1,9,9), dtype=dtype, device=device)
    mask_3[:,:,3:6,3:6] = 1.
    mask_d5 = torch.zeros((1,1,9,9), dtype=dtype, device=device)
    mask_d5[:,:,0:9:2,0:9:2] = 1.
    mask_d3 = torch.zeros((1,1,9,9), dtype=dtype, device=device)
    mask_d3[:,:,2:7:2,2:7:2] = 1.
    merge_kernel = w5 * (mask_5*a_5 + mask_3*a_3 + mask_d5*a_d5 + mask_d3*a_d3)

    if cout == cin:
        # avg
        wavg = torch.eye(cout, dtype=dtype, device=device).view(cout,cout,1)/9.
        wavg = wavg.repeat(1,1,9)
        wavg = wavg.view(cout,cout,3,3)
        wavg = torch.nn.functional.pad(wavg, (3,3,3,3), "constant", value=0)
        # skip-connect
        wskip = torch.eye(cout, dtype=dtype, device=device).view(cout,cout,1,1)
        wskip = torch.nn.functional.pad(wskip, (4,4,4,4), "constant", value=0)
    
        merge_kernel = merge_kernel + wavg*a_avg + wskip*a_skip + 0*a_none
    else:
        raise(ValueError("cout != cin, so skip-connect cannot be merged"))
        merge_kernel = w5_pad*a_5 + w3_pad*a_3 + dw3*a_d3 + dw5*a_d5 + 0*a_none
    return merge_kernel
    

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.act = nn.ReLU(inplace=False)
#    self.act = nn.SiLU()
#    self.act = nn.LeakyReLU(0.0, inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
#    x = self.act(x)
    x = torch.nn.functional.relu(x, inplace=False)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


