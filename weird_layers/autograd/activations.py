import torch

class MyReLUAutograd(torch.autograd.Function):
  '''Example activation'''
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.clamp(min=0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input
