
from torch import nn
from torchsummary import summary

class LeNetBlock(nn.Module):
  def __init__(self, iC, oC, kernel_size, act_mod=None, conv_kwargs=None,
               pool_kwargs=None):
    super(LeNetBlock, self).__init__()
    if act_mod is None:
      act_mod = nn.ReLU
    if conv_kwargs is None:
      conv_kwargs = {}
    if pool_kwargs is None:
      pool_kwargs = {}
    self.conv = nn.Conv2d(iC, oC, kernel_size, **conv_kwargs)
    self.pool = nn.AvgPool2d((2, 2), **pool_kwargs)
    self.act = act_mod()

  def forward(self, x):
    x = self.conv(x)
    x = self.pool(x)
    x = self.act(x)
    return x

class LeNet(nn.Module):
  def __init__(self, act_mod=None):
    super(LeNet, self).__init__()
    if act_mod is None:
      act_mod = nn.ReLU
    self.block1 = LeNetBlock(1, 6, (5, 5), act_mod,
                             conv_kwargs={'padding': 1})
    self.block2 = LeNetBlock(6, 16, (5, 5), act_mod,
                             conv_kwargs={'padding': 1},
                             pool_kwargs={'stride': 2})
    self.conv = nn.Conv2d(16, 120, (5, 5))
    self.act = act_mod()

    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(120, 84)
    self.fc1_act = act_mod()
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.conv(x)
    x = self.act(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc1_act(x)
    x = self.fc2(x)
    return x

if __name__ == '__main__':
  model = LeNet(nn.ReLU)
  summary(model, (1, 28, 28), device='cpu')
