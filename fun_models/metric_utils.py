import torch

def number_of_correct_metric(y_hat, y):
  _, predicted = torch.max(y_hat.data, 1)
  correct = (predicted == y).sum().item()
  return correct
