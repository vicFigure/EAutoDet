import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

from models.operations_merge import *


class MixedOp(nn.Module):

  def __init__(self, C, stride, single_sepConv=True, with_d5=True):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES_merge:
      op = OPS[primitive](C, stride, False, single_sepConv=single_sepConv, with_d5=with_d5)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    results = 0
    for idx, op in enumerate(self._ops):
       if 'merge' not in PRIMITIVES_merge[idx]:
           results += weights[idx]*op(x) 
       else:
           results += op(x, weights[idx:])
    return results


class Search_cell_merge(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, gumbel_op=False):
        super(Search_cell_merge, self).__init__()
        self.reduction = reduction
        self._multiplier = multiplier
        self.gumbel_op = gumbel_op

        if C_prev_prev is not None:
          self.num_input = 2
          if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
          else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        else: 
          self.preprocess0 = nn.Identity()
          self.num_input = 1
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
          for j in range(self.num_input+i):
            stride = 2 if reduction and j < self.num_input else 1
            op = MixedOp(C, stride)
            self._ops.append(op)
        self.final_act = nn.ReLU(inplace=False)
#        self.final_act = nn.SiLU()
#        self.final_act = Mish()

    def forward(self, inputs, weights):
        assert(len(inputs)<=2)
        if len(inputs)==2:
          s0, s1 = inputs
          s0 = self.preprocess0(s0)
          s1 = self.preprocess1(s1)
          states = [s0, s1]
        else:
          s1 = inputs[0]
          s1 = self.preprocess1(s1)
          states = [s1]

        offset = 0
        for i in range(self._steps):
          if self.gumbel_op:
            s = sum([self._ops[offset+j].forward_single(h, weights[offset+j]) for j, h in enumerate(states)])
          else:
            s = sum([self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states)])
          offset += len(states)
          states.append(s)
        return self.final_act(torch.cat(states[-self._multiplier:], dim=1))


