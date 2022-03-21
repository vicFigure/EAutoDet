import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

PRIMITIVES = [
#    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

#PRIMITIVES = [
##    'none',
#    'max_pool_3x3',
#    'avg_pool_3x3',
#    'skip_connect',
#    'conv_3x3',
#    'conv_5x5',
#    'conv_3x3_dil',
#    'conv_5x5_dil'
#]

OPS = {
  'noise': lambda C, stride, affine: NoiseOp(stride, 0., 1.),
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'merge_all_sep' : lambda C, stride, affine: MergeAllSep(C, C, 9, stride, 4, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_3x3' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine, dilation=1),
  'conv_5x5' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 2, affine=affine, dilation=1),
  'conv_3x3_dil' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 2, affine=affine, dilation=2),
  'conv_5x5_dil' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 4, affine=affine, dilation=2),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
#    Mish(),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
          x_new = x[:,:,::self.stride,::self.stride]
        else:
          x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
#        if self.training:
#          noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
#        else:
#          noise = torch.zeros_like(x_new)
        return noise


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, dilation=1, act=True):
    super(ReLUConvBN, self).__init__()
    self.act = act
    self.op = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.SiLU(),
#      Mish(),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
      )

  def forward(self, x):
    if self.act:
        x = torch.nn.functional.relu(x)
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.SiLU(),
#      Mish(),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    x = torch.nn.functional.relu(x)
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op1 = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.SiLU(),
#      Mish(),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      )
    self.op2 = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.SiLU(),
#      Mish(),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    x = torch.nn.functional.relu(x)
    x = self.op1(x)
    x = torch.nn.functional.relu(x)
    x = self.op2(x)
    return x


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

  def __init__(self, C_in, C_out, affine=True, act=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
#    self.act = nn.ReLU(inplace=False) if act else nn.Identity()
#    self.act = nn.SiLU() if act else nn.Identity()
#    self.act = Mish() if act else nn.Identity()
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    out = torch.nn.functional.relu(x)
    out = torch.cat([self.conv_1(out), self.conv_2(out[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


