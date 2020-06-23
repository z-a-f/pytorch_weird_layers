import torch
import unittest
from weird_layers import activations

class TestTrigonometric(unittest.TestCase):
  def test_sin(self):
    x = torch.randn(3, 4, 5)
    x.requires_grad = True
    y = activations.Sin()(x)
    loss = (y - 1).sum()
    loss.backward()
    self.assertFalse(x.grad is None)
