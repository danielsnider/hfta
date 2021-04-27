import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

try:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
except ImportError:
  pass


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.max_pool2d = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.max_pool2d(x)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

def train(config, model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    if config["device"] == 'xla':
      xm.optimizer_step(optimizer, barrier=True)
    else:
      optimizer.step()
    if batch_idx % config["log_interval"] == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch,
          batch_idx * len(data),
          len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss.item(),
      ))
      if config["dry_run"]:
        break


def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      N = target.size(0)
      output = model(data)
      test_loss += F.nll_loss(output, target,
                              reduction='none').view(-1, N).sum(dim=1)
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).view(-1, N).sum(dim=1)

  length = len(test_loader.dataset)
  test_loss /= length
  loss_str = ["%.4f" % e for e in test_loss]
  correct_str = [
      "%d/%d(%.2lf%%)" % (e, length, 100. * e / length) for e in correct
  ]
  print('Test set: \tAverage loss: {}, \n \t\t\tAccuracy: {}\n'.format(
      loss_str, correct_str))

def main(config):
  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  device = (torch.device(config["device"])
            if config["device"] in {'cpu', 'cuda'} else xm.xla_device())

  kwargs = {'batch_size': config["batch_size"]}
  kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))])

  dataset1 = datasets.MNIST('./data',
                            train=True,
                            download=True,
                            transform=transform)
  dataset2 = datasets.MNIST('./data', train=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

  model = Net().to(device)

  optimizer = optim.Adadelta(
      model.parameters(),
      lr=config["lr"][0],
  )

  start = time.perf_counter()
  for epoch in range(1, config["epochs"] + 1):
    now = time.perf_counter()
    train(config, model, device, train_loader, optimizer, epoch)
    print('Epoch {} took {} s!'.format(epoch, time.perf_counter() - now))
  end = time.perf_counter()

  test(model, device, test_loader)

  print('All jobs Finished, Each epoch took {} s on average!'.format(
      (end - start) / config["epochs"]))

config = {
    "device": "cuda",
    "batch_size": 64,
    "lr": [1.0],
    "gamma": 0.7,
    "epochs": 4,
    "seed": 1,
    "log_interval": 500,
    "dry_run": False,
    "save_model": False,
}

print(config)
main(config)
