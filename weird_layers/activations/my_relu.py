import torch

from ..autograd.activations import MyReLUAutograd

class MyReLU(torch.nn.Module):
  def forward(self, x):
    return MyReLUAutograd.apply(x)
