import json
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
from weird_layers import activations as trig

from run_utils import run_epoch
from log_utils import BatchPrintLogger

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
CPU_NUM = os.cpu_count()

EPOCHS = 30

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

DATA_PATH = os.path.join(os.path.expanduser('~'), 'data')
train_set = datasets.MNIST(root=DATA_PATH, train=True, transform=transform,
                           download=True)
test_set = datasets.MNIST(root=DATA_PATH, train=False, transform=transform,
                          download=True)

train_loader = DataLoader(train_set, batch_size=2048, shuffle=False,
                          num_workers=CPU_NUM)
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False,
                         num_workers=CPU_NUM)

models = {
  'ReLU': LeNet(act_mod=nn.ReLU).to(DEVICE),
  'Sin': LeNet(act_mod=trig.Sin).to(DEVICE),
  'Cos': LeNet(act_mod=trig.Cos).to(DEVICE),
  'TanH': LeNet(act_mod=nn.Tanh).to(DEVICE),
}

results = {}

for name, model in models.items():
  result = {
    'epochs': [],
    'train': {'accuracy': [], 'loss': []},
    'valid': {'accuracy': [], 'loss': []},
  }
  print(f'===> {name} <===')
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

  logger = BatchPrintLogger(print_every=10)

  for epoch in range(EPOCHS):
    result['epochs'].append(epoch+1)
    print(f'Epoch: {epoch+1}/{EPOCHS}')
    loss, accuracy, _ = run_epoch(model, train_loader, criterion,
                                  optimizer=optimizer, device=DEVICE,
                                  logger_fn=logger)
    result['train']['loss'].append(loss)
    result['train']['accuracy'].append(accuracy)
    loss, accuracy, _ = run_epoch(model, test_loader, criterion, device=DEVICE,
                                  logger_fn=logger)
    result['valid']['loss'].append(loss)
    result['valid']['accuracy'].append(accuracy)

  results[name] = result

this_file_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(this_file_path, 'results', 'lenet_trig.json')

with open(results_path, 'w') as fp:
  json.dump(results, fp)
