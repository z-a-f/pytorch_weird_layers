import torch
import unittest
from weird_layers.activations.my_relu import MyReLU

class TestMyReLU(unittest.TestCase):
  def test_constructor(self):
    my_relu = MyReLU()

  def test_forward_backward(self):
    x = torch.randn(3, 4, 5, 6)
    x.requires_grad = True
    my_relu = MyReLU()
    y = my_relu(x)
    loss = (y - 1).sum()
    loss.backward()
    self.assertFalse(x.grad is None)
