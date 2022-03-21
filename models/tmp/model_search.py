import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import numpy as np

class MixedOp_mergeAll(nn.Module):
  def __init__(self, C, stride, single_sepConv=True, with_d5=True):
    super(MixedOp_mergeAll, self).__init__()
    self._ops = nn.ModuleList()
    op = OPS['merge_all_sep'](C, stride, False)
    self._ops.append(op)

  def forward(self, x, weights):
    results = self._ops[0](x, weights)
    return results

class MixedOp(nn.Module):

  def __init__(self, C, stride, single_sepConv=True, with_d5=True):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for idx, primitive in enumerate(PRIMITIVES):
      if idx >= 4: break
      op = OPS[primitive](C, stride, False, single_sepConv=single_sepConv)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
    op = OPS['merge_conv_5x5'](C, stride, False, single_sepConv=single_sepConv, with_d5=with_d5)
#    op = OPS['merge_conv_normal'](C, stride, False, single_sepConv=single_sepConv, with_d5=with_d5)
#    op = OPS['merge_all_sep'](C, stride, False, single_sepConv=single_sepConv, with_d5=with_d5)
    self._ops.append(op)

  def forward(self, x, weights):
    results = 0
    for idx, op in enumerate(self._ops):
       if idx !=4:
           results += weights[idx]*op(x) 
       else:
#           tmp = op(x, weights[idx:])
#           print(results.shape, x.shape, tmp.shape)
#           results += tmp
           results += op(x, weights[idx:])
    return results


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
#        op = MixedOp_mergeAll(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None):
    super(Network, self).__init__()
    self.alpha_weights = alpha_weights
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if self.alpha_weights is None:  # for original darts
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      else:
        raise(ValueError("Why do you want to set alphas manually?"))
        print(self.alpha_weights['alphas_normal'])
        print(self.alpha_weights['alphas_reduce'])
        if cell.reduction:
          weights = self.alpha_weights['alphas_reduce']
        else:
          weights = self.alpha_weights['alphas_normal']
      
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def customized_weight(self, weights, numIn_per_node=2, choose_type='argmax'):
    """
    There are four types: argmax_operator, argmax_edge, sampe_operator, sample_edge
    In my opinion, the key should be argmax_edge & sample_edge

    argmax_operator: use argmax to choose the operator on each edge. But the input of each node is still full-sized.
    sample_operator: sample the operator on each edge. But the input of each node is still full-sized.

    argmax_edge: use argmax to choose the input for each node, default is 2, controlled by numIn_per_node
    sample_edge: sample the input for each node, default is 2, controlled by numIn_per_node
    """
    new_weights = torch.zeros_like(weights)
    weights_np = weights.data.cpu().numpy()
    if choose_type == 'argmax_edge':
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights_np[start:end].copy()
        actual_numIn = min(numIn_per_node, i+2)
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:actual_numIn]
        for j in edges:
          new_weights[start+j,:] = weights[start+j,:]
        start = end
        n += 1

    elif choose_type == 'sample_edge':
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights_np[start:end].copy()
        actual_numIn = min(numIn_per_node, i+2)
        p = np.max(W, axis=1)
        p = np.exp(p)
        p_sum = np.sum(p)
        p = p / p_sum
        edges = np.random.choice(range(i+2), actual_numIn, replace=False, p=p)
        for j in edges:
          new_weights[start+j,:] = weights[start+j,:]
        start = end
        n += 1

    elif choose_type == 'argmax_operator':
#      max_idx = np.argmax(weights_np, axis=1)
      max_idx = np.argpartition(-weights_np, 2, axis=1)[:,:2]
      for i in range(weights_np.shape[0]):  
        if max_idx[i][0] != PRIMITIVES.index('none'):
            new_weights[i, max_idx[i][0]] = weights[i, max_idx[i][0]]
        else:
            new_weights[i, max_idx[i][1]] = weights[i, max_idx[i][1]]

    elif choose_type == 'sample_operator':
      for i in range(weights_np.shape[0]):  
        idx = np.random.choice(range(weights_np.shape[1]), 2, replace=False, p=weights_np[i,:])
        if idx[0] != PRIMITIVES.index('none'):
            new_weights[i, idx[0]] = weights[i, idx[0]]
        else:
            new_weights[i, idx[1]] = weights[i, idx[1]]

    else:
      raise(ValueError('No type completed for type %s'%choose_type))
    return new_weights

  def customized_forward(self, input, numIn_per_node=2, choose_type='argmax_edge'):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)

      # get customized weights
      weights = self.customized_weight(weights, numIn_per_node=numIn_per_node, choose_type=choose_type)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

