import torch

class Sin(torch.nn.Module):
  def forward(self, inp):
    return torch.sin(inp)

class Cos(torch.nn.Module):
  def forward(self, inp):
    return torch.cos(inp)
