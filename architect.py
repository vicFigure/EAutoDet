import torch
from torch.cuda import amp
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils.torch_utils import is_parallel


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, compute_loss, accumulate, device, args, DDP=False):
#    self.network_momentum = args.momentum
#    self.network_weight_decay = args.weight_decay
    self.model = model
    self.ori_model = model.module if is_parallel(model) else model
#    if DDP: self.ori_model = self.model.module
#    else: self.ori_model = self.model
    self.device = device
    self.cuda = device.type != 'cpu'
    self.compute_loss = compute_loss
    self.scaler = amp.GradScaler(enabled=self.cuda)
    self.accumulate = accumulate
    self.ni = 0
    self.optimizer = torch.optim.Adam(self.ori_model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.optimizer.zero_grad()

  def step(self, input_valid, target_valid):
      self._backward_step(input_valid, target_valid)
      # Optimize
      if self.ni % self.accumulate == 0:
#          self.scaler.step(self.optimizer)  # optimizer.step
#          self.scaler.update()
          self.optimizer.step()
          self.optimizer.zero_grad()
          self.ni = 0

  def _backward_step(self, input_valid, target_valid):
    self.ni += 1
    with amp.autocast(enabled=self.cuda):
      pred = self.model(input_valid)  # forward
      loss, loss_items = self.compute_loss(pred, target_valid.to(self.device))  # loss scaled by batch_size
#    loss.backward() # it will affect model.parameters()
#    self.scaler.scale(loss).backward()
#    grads =  torch.autograd.grad(self.scaler.scale(loss), self.ori_model.arch_parameters(), grad_outputs=torch.ones_like(loss), allow_unused=True)
    grads =  torch.autograd.grad(loss, self.ori_model.arch_parameters(), grad_outputs=torch.ones_like(loss), allow_unused=True)
    for v, g in zip(self.ori_model.arch_parameters(), grads):
      if torch.isnan(g).any() or torch.isinf(g).any():
        print("gradient of architecture has NaN...")
        assert 0
        continue
      if v.grad is None:
        if not (g is None):
          v.grad = Variable(g.data)
      else:
        if not (g is None):
          v.grad.data.add_(g.data)

#    # Normalize the grad of channel alphas
#    search_space_per_layer = self.model.search_space_per_layer
#    for i, ch_alpha in enumerate(self.model.ch_arch_parameters()):
#      for e, g in zip(search_space_per_layer[i]["ch_ratio"], ch_alpha.grad):
#        g.data.mul_(1./e)


