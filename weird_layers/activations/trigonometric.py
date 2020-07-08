import torch
from torch import nn
from warnings import warn

class Sin(nn.Module):
  def forward(self, inp):
    return torch.sin(inp)

class Cos(nn.Module):
  def forward(self, inp):
    return torch.cos(inp)

class StackSinCos(nn.Module):
  '''Stacks the sin and cos as separate channels.

  Note: this changes the shape of the data.
  If the number of dimensions in the input = 2, and the channel_axis = 1,
  the input is reshaped into (?, 1, ?) and the layer is run on that.
  However, if reduce_channels is set, the result of stacking is pushed through a
  1x1 convolution to half the number of channels.

  If the input is (N, L) or (N, C, L): Conv1d is used
  If the input is (N, C, H, W): Conv2d is used
  if the input is (N, C, D, H, W): Conv3d is used
  '''
  def __init__(self, channel_axis, reduce_channels=False):
    super(StackSinCos, self).__init__()
    self.channel_axis = channel_axis
    self.reduce_channels = reduce_channels

    self.conv = nn.Conv1d(1, 1, 1)  # placeholder
    self.conv_set = False

  def forward(self, inp):
    original_shape = inp.shape
    if inp.ndim == 2 and self.channel_axis == 1:
      inp = inp.unsqueeze(1)
    if self.reduce_channels and not self.conv_set:
      self.conv_set = True
      ch = inp.shape[self.channel_axis]
      if inp.ndim <= 3:
        self.conv = nn.Conv1d(ch * 2, ch, 1)
      elif inp.ndim == 4:
        self.conv = nn.Conv2d(ch * 2, ch, 1)
      elif inp.ndim == 5:
        self.conv = nn.Conv3d(ch * 2, ch, 1)
      else:
        raise ValueError('Input malformed!')
      self.conv = self.conv.to(inp.device)

    assert self.channel_axis <= inp.ndim,\
      "Channel axis must be within the ndim of the input"
    y_sin = torch.sin(inp)
    y_cos = torch.cos(inp)

    x = torch.cat((y_sin, y_cos), dim=self.channel_axis)
    if self.reduce_channels:
      x = self.conv(x)
      x = x.reshape(original_shape)
    return x
