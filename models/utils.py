import torch
import torch.nn as nn

def sample_gumbel(shape, device, eps=1e-20):
    while True:
      gumbel = -torch.empty(shape, device=device).exponential_().log()
      if torch.isinf(gumbel).any() or torch.isnan(gumbel).any(): continue
      else: break
    return gumbel
#    U = torch.rand(shape, device=device)
#    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax_old(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

#useage: gumbel_softmax(F.log_softmax(alpha, dim=-1), hard=True))
def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    while True:
      y = gumbel_softmax_sample(logits, temperature)
      if torch.isinf(y).any() or torch.isnan(y).any(): continue
      else: break

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(-1, ind, 1.)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = y_hard - y.detach() + y
    return y_hard


def autopad(k, p=None, d=1):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = (k-1)*d // 2 if isinstance(k, int) else [(x-1)*d // 2 for x in k]  # auto-pad
    return p


