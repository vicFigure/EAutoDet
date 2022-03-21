# This file contains modules common to various models

import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list, plot_one_box
from models.darts_cell import Search_cell, Cell, PRIMITIVES
from models.mergenas_cell import Search_cell_merge
from models.utils import autopad, gumbel_softmax

#from mish_cuda import MishCuda as Mish



def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, d=1, s=s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, d=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(Conv, self).__init__()
        if isinstance(k, list): k = k[0]
        if isinstance(d, list): d = d[0]
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#        self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SepConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, d=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, dilation, stride, padding, groups
        super(SepConv, self).__init__()
        if isinstance(k, list): k = k[0]
        if isinstance(d, list): d = d[0]
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), dilation=d, groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, padding=0, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, k, d=1, shortcut=True, g=1, e=0.5, separable=False):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        if separable: my_conv = SepConv
        else: my_conv = Conv
        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv2 = my_conv(c_, c2, k, d=d, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Conv_search(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, kd=[(1,1), (3,1), (5,1), (3,2)], s=1, p=None, g=1, act=True, gumbel_op=False):
        super(Conv_search, self).__init__()
        self.gumbel_op = gumbel_op
        self.m = nn.ModuleList([])
        for ks in range(len(kd)):
          kernel_size = kd[ks][0]
          dilation = kd[ks][1]
          tmp = nn.Sequential(nn.Conv2d(c1, c2, kernel_size, s, autopad(kernel_size, p, dilation), dilation=dilation, groups=g, bias=False), nn.BatchNorm2d(c2))
          self.m.append(tmp)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#        self._alphas = torch.autograd.Variable(1e-3*torch.randn(len(k)), requires_grad=True)
        self.register_buffer('alphas', torch.autograd.Variable(1e-3*torch.randn(len(kd)), requires_grad=True))
        
    def forward(self, x):
        if self.gumbel_op: 
            alphas = gumbel_softmax(F.log_softmax(self.alphas, dim=-1), hard=True)
            index = alphas.max(-1, keepdim=True)[1].item()
            return self.act( alphas[index] * self.m[index](x) )
        else:
            alphas = F.softmax(self.alphas, dim=-1)
            return self.act(sum([a * m(x) for m, a in zip(self.m, alphas)]))

    def get_alphas(self):
        return [self.alphas]

    def get_op_alphas(self):
        return [self.alphas]

    def get_ch_alphas(self):
        return [None]

class Conv_search_merge(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, kd=[(1,1), (3,1), (5,1), (3,2)], candidate_e=[1.], s=1, p=None, g=1, act=True, gumbel_channel=False, with_bn=True, inside_alphas_channel=True, bias=False):
        # k=0 means zero op; d=0 means skip-connection
        super(Conv_search_merge, self).__init__()
        ks = 0
        for (k,d) in kd:
          if d == 0 and c1 != c2: raise(ValueError("d=0 means skip-connection, but c1 != c2."))
          if d==0 or k==0: continue
          tmp = (k-1)*d + 1
          if tmp > ks: ks = tmp
        self.cout = c2
        c2 = int(c2 * max(candidate_e))
        tmp1 = torch.Tensor(c2, c1, ks, ks)
        torch.nn.init.kaiming_normal_(tmp1,  mode='fan_in')
        self.weight = nn.Parameter(tmp1)
        self.kd = kd
        self.candidate_e = candidate_e
        self.padding = autopad(ks, None, 1)
        self.stride = s
        self.group = g
        gumbel_channel = gumbel_channel and len(candidate_e)>1
        self.gumbel_channel = gumbel_channel

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(c2*e)) for e in candidate_e]) if with_bn else nn.ModuleList([nn.Identity() for _ in candidate_e])
        else: self.bn = nn.BatchNorm2d(c2) if with_bn else nn.Identity()
        if bias: 
          b = torch.Tensor(c2)
          fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in)
          nn.init.uniform_(b, -bound, bound)
          self.bias = nn.Parameter(b)
        else: self.bias = None

#        self._alphas = torch.autograd.Variable(1e-3*torch.randn(len(k)), requires_grad=True)
        if len(kd) > 1:
          self.register_buffer('alphas', torch.autograd.Variable(1e-3*torch.randn(len(kd)), requires_grad=True))
        else: self.alphas = None
        if len(candidate_e) > 1 and inside_alphas_channel:
          self.register_buffer('alphas_channel', torch.autograd.Variable(1e-3*torch.randn(len(candidate_e)), requires_grad=True))
        else: self.alphas_channel = None

    def get_merge_kernel(self, w_base, alphas):
      Cout,C,ks,_ = w_base.shape
      merge_kernel = 0.
      for i, alpha in enumerate(alphas):
        k,d = self.kd[i]
        if k==0 or d==0: continue
        w = torch.zeros_like(w_base)
        tmp_ks = (k-1)*d + 1
        start = int((ks - tmp_ks) / 2)
        end = int(ks - start)
        w[:,:,start:end:d, start:end:d] = w_base[:,:,start:end:d, start:end:d]
        merge_kernel += w * alpha
      return merge_kernel
        
    def forward(self, x):
        Cin = x.size(1)
        bn = self.bn
        bias = self.bias
        if len(self.kd) > 1:
          alphas = nn.functional.softmax(self.alphas, dim=-1)
          merge_kernel = self.get_merge_kernel(self.weight, alphas)
        else:
          alphas = []
          merge_kernel = self.weight

        if len(self.candidate_e) > 1:
          Cout = merge_kernel.size(0)
          channel_mask = torch.zeros([Cout], dtype=merge_kernel.dtype, device=merge_kernel.device)
          if self.gumbel_channel:
            alphas_channel = gumbel_softmax(F.log_softmax(self.alphas_channel, dim=-1), hard=True) # alphas_channel is one-hot vector
            for idx, (e, a_e) in enumerate(zip(self.candidate_e, alphas_channel)):
              if a_e > 0.: 
                bn = self.bn[idx]
                if bias is not None: bias = bias[:int(self.cout*e)] 
                merge_kernel = merge_kernel[:int(self.cout*e),:Cin,:,:] * a_e
                break
          else:
            channel_idx = torch.arange(0, Cout, dtype=merge_kernel.dtype, device=merge_kernel.device).long()
  #          channel_idx = torch.sort(merge_kernel.view(Cout,-1).sum(dim=-1), descending=True)[1]
            alphas_channel = nn.functional.softmax(self.alphas_channel, dim=-1)
            for e, a_e in zip(self.candidate_e, alphas_channel):
              channel_mask[channel_idx[:int(e*self.cout)]] += a_e
            merge_kernel = merge_kernel * channel_mask.view(-1,1,1,1)
        if Cin != merge_kernel.size(1): merge_kernel = merge_kernel[:,:Cin,:,:]
        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.group)
        
        for i, alpha in enumerate(alphas):
          k,d = self.kd[i]
          if k==0: out = out + 0.*alpha
          elif d == 0: out = out + x*alpha
        if bias is None: return self.act(bn(out))
        else: return self.act(bn(out)) + bias.view(1,-1,1,1)

    def forward_withAlpha(self, x, alphas_channel):
        assert len(self.candidate_e)==len(alphas_channel)
        if len(self.kd) > 1:
          alphas = nn.functional.softmax(self.alphas, dim=-1)
          merge_kernel = self.get_merge_kernel(self.weight, alphas)
        else:
          alphas = []
          merge_kernel = self.weight

        Cout = merge_kernel.size(0)
        channel_mask = torch.zeros([Cout], dtype=merge_kernel.dtype, device=merge_kernel.device)
        if self.gumbel_channel:
          Cin = x.size(1)
          for idx, (e, a_e) in enumerate(zip(self.candidate_e, alphas_channel)):
            if a_e > 0.: bn = self.bn[idx]; merge_kernel = merge_kernel[:int(self.cout*e),:Cin,:,:] * a_e
        else:
          bn = self.bn
          channel_idx = torch.arange(0, Cout, dtype=merge_kernel.dtype, device=merge_kernel.device).long()
  #        channel_idx = torch.sort(merge_kernel.view(Cout,-1).sum(dim=-1), descending=True)[1]
          for e, a_e in zip(self.candidate_e, alphas_channel):
            channel_mask[channel_idx[:int(e*self.cout)]] += a_e
          merge_kernel = merge_kernel * channel_mask.view(-1,1,1,1)
        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.group)

        for i, alpha in enumerate(alphas):
          k,d = self.kd[i]
          if k==0: out = out + 0.*alpha
          elif d == 0: out = out + x*alpha
        return self.act(bn(out))

    def get_alphas(self):
        out = []
        if self.alphas is not None: out.append(self.alphas)
        if self.alphas_channel is not None: out.append(self.alphas_channel)
        return out

    def get_op_alphas(self):
        return [self.alphas]

    def get_ch_alphas(self):
        return [self.alphas_channel]

class SepConv_search_merge(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, kd=[(1,1), (3,1), (5,1), (3,2)], candidate_e=[1.], s=1, p=None, g=1, act=True, gumbel_channel=False, with_bn=True, inside_alphas_channel=True, bias=False):
        # k=0 means zero op; d=0 means skip-connection
        super(SepConv_search_merge, self).__init__()
        ks = 0
        for (k,d) in kd:
          if d == 0 and c1 != c2: raise(ValueError("d=0 means skip-connection, but c1 != c2."))
          if d==0 or k==0: continue
          tmp = (k-1)*d + 1
          if tmp > ks: ks = tmp
        self.cout = c2
        c2 = int(c2 * max(candidate_e))
        tmp1 = torch.Tensor(c1, 1, ks, ks)
        torch.nn.init.kaiming_normal_(tmp1,  mode='fan_in')
        self.depth_weight = nn.Parameter(tmp1)
        tmp2 = torch.Tensor(c2, c1, 1, 1)
        torch.nn.init.kaiming_normal_(tmp2,  mode='fan_in')
        self.point_weight = nn.Parameter(tmp2)
        self.kd = kd
        self.candidate_e = candidate_e
        self.padding = autopad(ks, None, 1)
        self.stride = s
        self.group = g
        gumbel_channel = gumbel_channel and len(candidate_e)>1
        self.gumbel_channel = gumbel_channel

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if gumbel_channel: self.bn = nn.ModuleList([nn.BatchNorm2d(int(c2*e)) for e in candidate_e]) if with_bn else nn.ModuleList([nn.Identity() for _ in candidate_e])
        else: self.bn = nn.BatchNorm2d(c2) if with_bn else nn.Identity()
        if bias: 
          b = torch.Tensor(c2)
          fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in)
          nn.init.uniform_(b, -bound, bound)
          self.bias = nn.Parameter(b)
        else: self.bias = None

#        self._alphas = torch.autograd.Variable(1e-3*torch.randn(len(k)), requires_grad=True)
        if len(kd) > 1:
          self.register_buffer('alphas', torch.autograd.Variable(1e-3*torch.randn(len(kd)), requires_grad=True))
        else: self.alphas = None
        if len(candidate_e) > 1 and inside_alphas_channel:
          self.register_buffer('alphas_channel', torch.autograd.Variable(1e-3*torch.randn(len(candidate_e)), requires_grad=True))
        else: self.alphas_channel = None

    def get_merge_kernel(self, w_base, alphas):
      Cout,C,ks,_ = w_base.shape
      merge_kernel = 0.
      for i, alpha in enumerate(alphas):
        k,d = self.kd[i]
        if k==0 or d==0: continue
        w = torch.zeros_like(w_base)
        tmp_ks = (k-1)*d + 1
        start = int((ks - tmp_ks) / 2)
        end = int(ks - start)
        w[:,:,start:end:d, start:end:d] = w_base[:,:,start:end:d, start:end:d]
        merge_kernel += w * alpha
      return merge_kernel
        
    def forward(self, x):
        Cin = x.size(1)
        bn = self.bn
        bias = self.bias
        # kernel size for depth-wise conv
        if len(self.kd) > 1:
          alphas = nn.functional.softmax(self.alphas, dim=-1)
          merge_kernel = self.get_merge_kernel(self.depth_weight, alphas)
        else:
          alphas = []
          merge_kernel = self.depth_weight
        if Cin != merge_kernel.size(0): merge_kernel = merge_kernel[:Cin,:,:,:]
        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)

        # out channel for point-wise conv
        pweight = self.point_weight
        if len(self.candidate_e) > 1:
          Cout = pweight.size(0)
          if self.gumbel_channel:
            alphas_channel = gumbel_softmax(F.log_softmax(self.alphas_channel, dim=-1), hard=True) # alphas_channel is one-hot vector
            for idx, (e, a_e) in enumerate(zip(self.candidate_e, alphas_channel)):
              if a_e > 0.: 
                bn = self.bn[idx]
                if bias is not None: bias = bias[:int(self.cout*e)] 
                pweight = pweight[:int(self.cout*e),:Cin,:,:] * a_e
                break
          else:
            channel_mask = torch.zeros([Cout], dtype=pweight.dtype, device=pweight.device)
            channel_idx = torch.arange(0, Cout, dtype=pweight.dtype, device=pweight.device).long()
  #          channel_idx = torch.sort(pweight.view(Cout,-1).sum(dim=-1), descending=True)[1]
            alphas_channel = nn.functional.softmax(self.alphas_channel, dim=-1)
            for e, a_e in zip(self.candidate_e, alphas_channel):
              channel_mask[channel_idx[:int(e*self.cout)]] += a_e
            pweight = pweight * channel_mask.view(-1,1,1,1)
        if Cin != pweight.size(1): pweight = pweight[:,:Cin,:,:]
        out = torch.nn.functional.conv2d(out, pweight, stride=1, padding=0, dilation=1, groups=self.group)
        
        for i, alpha in enumerate(alphas):
          k,d = self.kd[i]
          if k==0: out = out + 0.*alpha
          elif d == 0: out = out + x*alpha
        if bias is None: return self.act(bn(out))
        else: return self.act(bn(out)) + bias.view(1,-1,1,1)

    def forward_withAlpha(self, x, alphas_channel):
        assert len(self.candidate_e)==len(alphas_channel)
        bias = self.bias
        Cin = x.size(1)
        # kernel size for depth-wise conv
        if len(self.kd) > 1:
          alphas = nn.functional.softmax(self.alphas, dim=-1)
          merge_kernel = self.get_merge_kernel(self.depth_weight, alphas)
        else:
          alphas = []
          merge_kernel = self.depth_weight
        if Cin != merge_kernel.size(0): merge_kernel = merge_kernel[:Cin,:,:,:]
        out = torch.nn.functional.conv2d(x, merge_kernel, stride=self.stride, padding=self.padding, dilation=1, groups=Cin)

        # out channel for point-wise conv
        pweight = self.point_weight
        Cout = pweight.size(0)
        if self.gumbel_channel:
          for idx, (e, a_e) in enumerate(zip(self.candidate_e, alphas_channel)):
            if a_e > 0.: 
              bn = self.bn[idx]
              if bias is not None: bias = bias[:int(self.cout*e)] 
              pweight = pweight[:int(self.cout*e),:Cin,:,:] * a_e
              break
#            if a_e > 0.: bn = self.bn[idx]; pweight = pweight[:int(self.cout*e),:Cin,:,:] * a_e
        else:
          bn = self.bn
          channel_mask = torch.zeros([Cout], dtype=pweight.dtype, device=pweight.device)
          channel_idx = torch.arange(0, Cout, dtype=pweight.dtype, device=pweight.device).long()
  #        channel_idx = torch.sort(pweight.view(Cout,-1).sum(dim=-1), descending=True)[1]
          for e, a_e in zip(self.candidate_e, alphas_channel):
            channel_mask[channel_idx[:int(e*self.cout)]] += a_e
          pweight = pweight * channel_mask.view(-1,1,1,1)
        if Cin != pweight.size(1): pweight = pweight[:,:Cin,:,:]
        out = torch.nn.functional.conv2d(out, pweight, stride=1, padding=0, dilation=1, groups=self.group)

        for i, alpha in enumerate(alphas):
          k,d = self.kd[i]
          if k==0: out = out + 0.*alpha
          elif d == 0: out = out + x*alpha
        if bias is None: return self.act(bn(out))
        else: return self.act(bn(out)) + bias.view(1,-1,1,1)

    def get_alphas(self):
        out = []
        if self.alphas is not None: out.append(self.alphas)
        if self.alphas_channel is not None: out.append(self.alphas_channel)
        return out

    def get_op_alphas(self):
        return [self.alphas]

    def get_ch_alphas(self):
        return [self.alphas_channel]

class Bottleneck_search(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, kd=[(3,1),(5,1),(3,2)], shortcut=True, g=1, e=0.5, gumbel_op=False):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_search, self).__init__()
        self.gumbel = gumbel_op
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv2 = Conv_search(c_, c2, kd, 1, g=g, gumbel_op=gumbel_op)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    def get_alphas(self):
        return self.cv2.get_alphas()

    def get_op_alphas(self):
        return self.cv2.get_op_alphas()

    def get_ch_alphas(self):
        return self.cv2.get_ch_alphas()

class Bottleneck_search_merge(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, kd=[(3,1),(5,1),(3,2)], candidate_e=[1.], shortcut=True, g=1, e=0.5, gumbel_channel=False, separable=False):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_search_merge, self).__init__()
        self.gumbel_channel = gumbel_channel

        c_ = int(c2 * e)  # hidden channels
        c_max = int(c_ * max(candidate_e))
#        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv1 = Conv_search_merge(c1, c_max, kd=[(1,1)], candidate_e=candidate_e, s=1, gumbel_channel=gumbel_channel)
        if separable: my_conv = SepConv_search_merge
        else: my_conv = Conv_search_merge
        self.cv2 = my_conv(c_max, c2, kd, candidate_e=[1.], s=1, g=g, gumbel_channel=gumbel_channel)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.gumbel_channel:
          cout = x.size(1)
          out = self.cv2(self.cv1(x))
          return x + out[:,:cout,:,:] if self.add else out
        else:
          return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    def get_alphas(self):
        out = self.cv1.get_alphas()
        out.extend(self.cv2.get_alphas())
        return out

    def get_op_alphas(self):
        return self.cv2.get_op_alphas()

    def get_ch_alphas(self):
        return self.cv1.get_ch_alphas()


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1, d=1, s=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, k=3, d=1, shortcut=True, g=1, e=0.5, e_bottleneck=1., separable=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        if isinstance(k, int): ks = [k for _ in range(n)]
        else: ks = k
        if isinstance(d, int): ds = [d for _ in range(n)]
        else: ds = d
        if isinstance(e_bottleneck, float): es = [e_bottleneck for _ in range(n)]
        else: es = e_bottleneck
        assert len(ks) >= n
        assert len(ds) >= n
        assert (len(es) >= n) or (len(es) >= n+1)

        if isinstance(c2, int): c2=[c2 for _ in range(n)]  
        c1out = int(c2[0]*e); c2out = int(c2[-1]*e)  # hidden channels

        self.cv1 = Conv(c1, c1out, k=1, d=1, s=1)
        m_list = []; cin = c1out
        for i in range(n):
          m_list.append(Bottleneck(cin, int(c2[i]*e), ks[i], ds[i], shortcut, g, e=es[i], separable=separable))
          cin = int(c2[i]*e)
        self.m = nn.Sequential(*m_list)
#        self.m = nn.Sequential(*[Bottleneck(c_, c_, ks[i], ds[i], shortcut, g, e=1.0) for i in range(n)])
#        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.cv2 = Conv(c1, c2out, k=1, d=1, s=1)
        self.cv3 = Conv(2 * c2out, c2[-1], k=1, d=1, s=1)  # act=FReLU(c2)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3_search(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, kd=[(3,1),(5,1),(3,2)], shortcut=True, g=1, e=0.5, gumbel_op=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_search, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv3 = Conv(2 * c_, c2, k=1, d=1, s=1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck_search(c_, c_, kd, shortcut, g, e=1.0, gumbel_op=gumbel_op) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

    def get_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_alphas())
        return alphas

    def get_op_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_op_alphas())
        return alphas

    def get_ch_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_ch_alphas())
        return alphas

class C3_search_merge(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, kd=[(3,1),(5,1),(3,2)], candidate_e=[1.], shortcut=True, g=1, e=0.5, search_c2=None, gumbel_channel=False, separable=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_search_merge, self).__init__()
        if search_c2==True:
          self.search_c2 = candidate_e
        else:
          self.search_c2 = search_c2
        self.gumbel_channel = gumbel_channel

        if self.search_c2:
          c2 = c2 * max(self.search_c2)
          c_ = int(c2 * e)  # hidden channels
          self.cv1 = Conv_search_merge(c1, c_, kd=[(1,1)], candidate_e=self.search_c2, s=1, gumbel_channel=gumbel_channel, inside_alphas_channel=False)
          self.cv2 = Conv_search_merge(c1, c_, kd=[(1,1)], candidate_e=self.search_c2, s=1, gumbel_channel=gumbel_channel, inside_alphas_channel=False)
          if gumbel_channel:
            self.cv3 = nn.ModuleList([Conv_search_merge(c_, c2, kd=[(1,1)], candidate_e=self.search_c2, s=1, gumbel_channel=gumbel_channel, act=False, with_bn=False, inside_alphas_channel=False) for _ in range(2)])  # act=FReLU(c2)
            self.cv3_bn = nn.ModuleList([nn.BatchNorm2d(int(c2*e)) for e in self.search_c2])
            self.cv3_act = nn.SiLU()
          else:
            self.cv3 = Conv_search_merge(2 * c_, c2, kd=[(1,1)], candidate_e=self.search_c2, s=1, gumbel_channel=gumbel_channel, inside_alphas_channel=False)  # act=FReLU(c2)
          self.register_buffer('alphas_channel', torch.autograd.Variable(1e-3*torch.randn(len(self.search_c2)), requires_grad=True))
        else:
          c_ = int(c2 * e)  # hidden channels
          self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
          self.cv2 = Conv(c1, c_, k=1, d=1, s=1)
          self.cv3 = Conv(2 * c_, c2, k=1, d=1, s=1)  # act=FReLU(c2)
          self.alphas_channel = None
        self.m = nn.Sequential(*[Bottleneck_search_merge(c_, c_, kd, candidate_e, shortcut, g, e=1.0, gumbel_channel=gumbel_channel, separable=separable) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        if self.search_c2:
          if self.gumbel_channel:
            alphas_channel = gumbel_softmax(F.log_softmax(self.alphas_channel, dim=-1), hard=True) # alphas_channel is one-hot vector
            out = self.cv3[0].forward_withAlpha(self.m(self.cv1.forward_withAlpha(x, alphas_channel)), alphas_channel) + self.cv3[1].forward_withAlpha(self.cv2.forward_withAlpha(x, alphas_channel), alphas_channel)
            for idx, a in enumerate(alphas_channel):
              if a > 0: return self.cv3_act(self.cv3_bn[idx](out))
            raise(ValueError("If code runs here, then there must be something wrong!"))
          else:
            alphas_channel = nn.functional.softmax(self.alphas_channel, dim=-1)
            return self.cv3.forward_withAlpha(torch.cat((self.m(self.cv1.forward_withAlpha(x, alphas_channel)), self.cv2.forward_withAlpha(x, alphas_channel)), dim=1), alphas_channel)
        else:
          return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

    def get_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_alphas())
        if self.search_c2: alphas.append(self.alphas_channel)
        return alphas

    def get_op_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_op_alphas())
        return alphas

    def get_ch_alphas(self):
        alphas = []
        for m in self.m:
          alphas.extend(m.get_ch_alphas())
        if self.search_c2: alphas.append(self.alphas_channel)
        return alphas

class Cells_search(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, N=1, gumbel_op=False):
        super(Cells_search, self).__init__()
        self._steps = steps
        self.gumbel_op = gumbel_op
        self.cells = nn.ModuleList()
        C_curr = C
        self.num_input = 1 if C_prev_prev==None else 2
        for i in range(N):
          cell = Search_cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, gumbel_op)
          C_prev_prev, C_prev = C_prev, multiplier*C_curr
          if self.num_input==1: C_prev_prev = None
          reduction_prev = reduction
          reduction = False
          self.cells.append(cell)

        k = sum(1 for i in range(self._steps) for n in range(self.num_input+i))
        self.register_buffer('alphas', torch.autograd.Variable(1e-3*torch.randn(k, len(PRIMITIVES)), requires_grad=True))

    def forward(self, x):
        assert(len(x)==self.num_input)
        if self.gumbel_op: weights = gumbel_softmax(F.log_softmax(self.alphas, dim=-1), hard=True)
        else: weights = F.softmax(self.alphas, dim=-1)
        if self.num_input == 2:
          s0, s1 = x[:2]
          for cell in self.cells:
            s0, s1 = s1, cell([s0,s1], weights)
        else:
          s1 = x[-1]
          for cell in self.cells:
            s1 = cell([s1], weights)
            
        return s1

    def get_alphas(self):
        alphas = self.get_op_alphas()
        return alphas

    def get_op_alphas(self):
        alphas = [self.alphas]
        return alphas

    def get_ch_alphas(self):
        return [None]

class Cells_search_merge(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, N=1, gumbel_op=False):
        super(Cells_search_merge, self).__init__()
        self._steps = steps
        self.gumbel_op = gumbel_op
        self.cells = nn.ModuleList()
        C_curr = C
        self.num_input = 1 if C_prev_prev==None else 2
        for i in range(N):
          cell = Search_cell_merge(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, gumbel_op)
          C_prev_prev, C_prev = C_prev, multiplier*C_curr
          if self.num_input==1: C_prev_prev = None
          reduction_prev = reduction
          reduction = False
          self.cells.append(cell)

        k = sum(1 for i in range(self._steps) for n in range(self.num_input+i))
        self.register_buffer('alphas', torch.autograd.Variable(1e-3*torch.randn(k, len(PRIMITIVES)), requires_grad=True))

    def forward(self, x):
        assert(len(x)==self.num_input)
        if self.gumbel_op: weights = gumbel_softmax(F.log_softmax(self.alphas, dim=-1), hard=True)
        else: weights = F.softmax(self.alphas, dim=-1)
        if self.num_input == 2:
#          s0, s1 = x[:2]
          for cell in self.cells:
#            s0, s1 = s1, cell([s0,s1], weights)
             s1 = cell(x[-2:], weights)
             x.append(s1)
        else:
          s1 = x[-1]
          for cell in self.cells:
            s1 = cell([s1], weights)
            
        return s1

    def get_alphas(self):
        alphas = self.get_op_alphas()
        return alphas

    def get_op_alphas(self):
        alphas = [self.alphas]
        return alphas

    def get_ch_alphas(self):
        return [None]
    
class Cells(nn.Module):
    def __init__(self, genotype, concat, C_prev_prev, C_prev, C, reduction, reduction_prev, N=1, drop_path_prob=0.0):
        super(Cells, self).__init__()
        self.drop_path_prob = drop_path_prob
        self.cells = nn.ModuleList()
        C_curr = C
        self.num_input = 1 if C_prev_prev==None else 2
        for i in range(N):
          cell = Cell(genotype, concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
          C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
          if self.num_input==1: C_prev_prev = None
          reduction_prev = reduction
          reduction = False
          self.cells.append(cell)

    def forward(self, x):
        assert(len(x)==self.num_input)
        if self.num_input == 2:
          s0, s1 = x[:2]
          for cell in self.cells:
            s0, s1 = s1, cell([s0,s1], self.drop_path_prob)
        else:
          s1 = x[-1]
          for cell in self.cells:
            s1 = cell([s1], self.drop_path_prob)
        return s1

class AFF(nn.Module):
    # Auto-Feature Fusion
    def __init__(self, c1s, c2, strides, kd=[(1,1),(3,1),(5,1),(3,2)], candidate_e=[1.], gumbel_channel=False, separable=False):
        """
        strides: a list indicating the scale for each edge. Whether to up-sampling or down-sampling, and how much the degree is
        """
        super(AFF, self).__init__()
        assert(len(c1s)==len(strides))
        self.cin = c1s
        self.cout = c2
        self.strides = strides
        self.candidate_e = candidate_e
        self.gumbel_channel = gumbel_channel
        self.m = nn.ModuleList([])
        self.up_sample = nn.ModuleList([])
        if separable: my_conv = SepConv_search_merge
        else: my_conv = Conv_search_merge
        for c, s in zip(c1s, strides):
#          if c==c2: tmp_kd = [(-1,0)] + kd # add skip-connection to the search space
#          else: tmp_kd = kd
          tmp_kd = kd
          if s < 1: 
            self.m.append(my_conv(c, c2, kd=tmp_kd, candidate_e=candidate_e, s=1, gumbel_channel=gumbel_channel, inside_alphas_channel=False))
            self.up_sample.append(nn.Upsample(None, int(1./s), mode='nearest'))
          else:
            self.m.append(my_conv(c, c2, kd=tmp_kd, candidate_e=candidate_e, s=s, gumbel_channel=gumbel_channel, inside_alphas_channel=False))
            self.up_sample.append(nn.Identity())
        self.register_buffer('alphas_edge', torch.autograd.Variable(1e-3*torch.randn(len(c1s)), requires_grad=True)) # architecture parameters for super-edges
        if len(self.candidate_e)>1:
          self.register_buffer('alphas_channel', torch.autograd.Variable(1e-3*torch.randn(len(self.candidate_e)), requires_grad=True))
        else: self.alphas_channel = None

    def forward(self, xs):
       alphas_edge = nn.functional.softmax(self.alphas_edge, dim=-1)
       out = 0.
       if self.alphas_channel is not None:
         if self.gumbel_channel:
           alphas_channel = gumbel_softmax(F.log_softmax(self.alphas_channel, dim=-1), hard=True) # alphas_channel is one-hot vector
         else:
           alphas_channel = nn.functional.softmax(self.alphas_channel, dim=-1)
         for idx, (m, up, x) in enumerate(zip(self.m, self.up_sample, xs)):
           out += up(m.forward_withAlpha(x, alphas_channel)) * alphas_edge[idx]
       else:
         for idx, (m, up, x) in enumerate(zip(self.m, self.up_sample, xs)):
           out += up(m(x)) * alphas_edge[idx]
       return out
 
    def get_alphas(self):
        out = [self.alphas_edge]
        if self.alphas_channel is not None: out.append(self.alphas_channel)
        for m in self.m:
          out.extend(m.get_alphas())
        return out

    def get_op_alphas(self):
        out = []
        for m in self.m:
          out.extend(m.get_op_alphas())
        return out

    def get_ch_alphas(self):
        out = []
        for m in self.m:
          out.extend(m.get_ch_alphas())
        out.append(self.alphas_channel)
        return out

    def get_edge_alphas(self):
        return [self.alphas_edge]
        
class FF(nn.Module):
    # Auto-Feature Fusion
    def __init__(self, c1s, c2, strides, ks, ds, separable=False):
        super(FF, self).__init__()
        assert(len(c1s)==len(strides))
        assert(len(c1s)==len(ks))
        assert(len(c1s)==len(ds))
        self.cin = c1s
        self.cout = c2
        self.strides = strides
        self.m = nn.ModuleList([])
        self.up_sample = nn.ModuleList([])
        if separable: my_conv = SepConv
        else: my_conv = Conv
        for c, s, k, d in zip(c1s, strides, ks, ds):
          if k == 0: # zero op
            tmp_m = None; tmp_up = None;
          elif d == 0: # skip-connection op
            tmp_m = nn.Identity(); tmp_up = nn.Identity();
          else:
            if s < 1:
              tmp_m = my_conv(c, c2, k, d, s=1)
              tmp_up = nn.Upsample(None, int(1./s), mode='nearest')
            else:
              tmp_m = my_conv(c, c2, k, d, s=s)
              tmp_up = nn.Identity()
          self.m.append(tmp_m)
          self.up_sample.append(tmp_up)
#          if k == 0: self.m.append(None); self.up_sample.append(None); continue; # zero op
#          elif d == 0: self.m.append(nn.Identity()); self.up_sample.append(nn.Identity()); continue; # skip
#          if s < 1: 
#            self.m.append(my_conv(c, c2, k, d, s=1))
#            self.up_sample.append(nn.Upsample(None, int(1./s), mode='nearest'))
#          else:
#            self.m.append(my_conv(c, c2, k, d, s=s))
#            self.up_sample.append(nn.Identity())

    def forward(self, xs):
       out = 0.
       for idx, (m, up, x) in enumerate(zip(self.m, self.up_sample, xs)):
         if m is not None:
           out += up(m(x))
       return out


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, k=1, d=1, s=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPP_search(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP_search, self).__init__()
        c_ = c1 // 2  # hidden channels
#        self.cv1 = Conv(c1, c_, k=1, d=1, s=1)
        self.cv1 = Conv_search_merge(c1, c_, kd=[(1,1)], candidate_e=[1.], s=1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, k=1, d=1, s=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, files, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = Path(save_dir) / self.files[i]
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='results/'):
        Path(save_dir).mkdir(exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

