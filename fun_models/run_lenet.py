import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from lenet import LeNet
from weird_layers.activations import Sin

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
CPU_NUM = os.cpu_count()

EPOCHS = 30

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

DATA_PATH = os.path.join(os.path.expanduser('~'), 'data')
train_set = datasets.MNIST(root=DATA_PATH, train=True, transform=transform, download=True)
test_set = datasets.MNIST(root=DATA_PATH, train=False, transform=transform, download=True)

train_loader = DataLoader(train_set, batch_size=2048, shuffle=False, num_workers=CPU_NUM)
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=CPU_NUM)

model = LeNet(act_mod=Sin).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
writer = SummaryWriter()

for epoch in range(EPOCHS):
  running_loss = 0.0
  running_acc = 0.0
  total_samples = 0

  model.train()
  for idx, data in enumerate(train_loader, 0):
    x, y = data
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    optimizer.zero_grad()

    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(y_hat.data, 1)
    running_acc += (predicted == y).sum().item()

    running_loss += loss.item()
    total_samples += len(x)
  writer.add_scalar('Loss/train', running_loss / total_samples, epoch)
  writer.add_scalar('Accuracy/train', running_acc / total_samples, epoch)

  running_loss = 0.0
  running_acc = 0.0
  total_samples = 0
  model.eval()
  with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
      x, y = data
      x = x.to(DEVICE)
      y = y.to(DEVICE)

      y_hat = model(x)
      loss = criterion(y_hat, y)

      _, predicted = torch.max(y_hat.data, 1)
      running_acc += (predicted == y).sum().item()

      running_loss += loss.item()
      total_samples += len(x)
  writer.add_scalar('Loss/test', running_loss / total_samples, epoch)
  writer.add_scalar('Accuracy/test', running_acc / total_samples, epoch)
  print('.', end='')
print()
