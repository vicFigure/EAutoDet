import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mish_cuda import MishCuda as Mish

from models.operations import *


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

  def forward_single(self, x, weights):
    index = weights.max(-1, keepdim=True)[1].item()
    return weights[index] * self._ops[index](x)

class Search_cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, gumbel_op=False):
        super(Search_cell, self).__init__()
        self.reduction = reduction
        self._multiplier = multiplier
        self.gumbel_op = gumbel_op

        self.preprocess0 = None
        if C_prev_prev is not None:
          self.num_input = 2
          if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False, act=True)
          else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, act=True)
        else: 
          self.num_input = 1
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, act=True)
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
#        return torch.cat(states[-self._multiplier:], dim=1)
        return self.final_act(torch.cat(states[-self._multiplier:], dim=1))

class Cell(nn.Module):

  def __init__(self, genotype, concat, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if C_prev_prev is not None:
      self.num_input = 2
      if reduction_prev:
        self.preprocess0 = FactorizedReduce(C_prev_prev, C, act=True)
      else:
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, act=True)
    else:
      self.preprocess0 = None
      self.num_input = 1
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, act=True)
    
    op_names, indices = zip(*genotype)
    concat = concat
    self._compile(C, op_names, indices, concat, reduction)
    self.final_act = nn.ReLU(inplace=False)
#    self.final_act = Mish()

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < self.num_input else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, inputs, drop_prob=0.0):
    assert(len(inputs)<=2)
    if len(inputs)==2:
      s0, s1 = inputs
      s0 = self.preprocess0(s0)
      s1 = self.preprocess1(s1)
      states = [s0, s1]
      for i in range(self._steps):
        h1 = states[self._indices[2*i]]
        h2 = states[self._indices[2*i+1]]
        op1 = self._ops[2*i]
        op2 = self._ops[2*i+1]
        h1 = op1(h1)
        h2 = op2(h2)
        if self.training and drop_prob > 0.:
          if not isinstance(op1, Identity):
            h1 = drop_path(h1, drop_prob)
          if not isinstance(op2, Identity):
            h2 = drop_path(h2, drop_prob)
        s = h1 + h2
        states += [s]
    else:
      s1 = inputs[0]
      s1 = self.preprocess1(s1)
      states = [s1]
      for i in range(self._steps):
        h1 = states[self._indices[i]]
        op1 = self._ops[i]
        h1 = op1(h1)
        if self.training and drop_prob > 0.:
          if not isinstance(op1, Identity):
            h1 = drop_path(h1, drop_prob)
        s = h1
        states += [s]

    return self.final_act(torch.cat([states[i] for i in self._concat], dim=1))

def genotype(alphas, steps, multiplier, num_input=2):
    def _parse(weights):
      gene = []
      n = num_input
      start = 0
      try:
        none_idx = PRIMITIVES.index('none')
      except:
        none_idx = -1
      for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + num_input), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_idx))[:num_input]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != none_idx:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    geno = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy())

    concat = list(range(2+steps-multiplier, steps+2))
    return geno, concat

