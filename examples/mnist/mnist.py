# From https://colab.research.google.com/github/UofT-EcoSystem/hfta/blob/eric/colab-tutorial/docs/HFTA_PyTorch_Tutorial.ipynb#scrollTo=N44NF4HoalOh
from __future__ import print_function
import sys
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

# Use helper functions from hfta package to convert your operators and optimizors
from hfta.ops import get_hfta_op_for
from hfta.optim import get_hfta_optim_for


class Net(nn.Module):

  # When initializing the model, save the number of fused models (B),
  # and convert the default operators to HFTA version with get_hfta_op_for(<default>, B).
  def __init__(self, B=0):
    super(Net, self).__init__()
    self.B = B
    self.conv1 = get_hfta_op_for(nn.Conv2d, B=B)(1, 32, 3, 1)
    self.conv2 = get_hfta_op_for(nn.Conv2d, B=B)(32, 64, 3, 1)
    self.max_pool2d = get_hfta_op_for(nn.MaxPool2d, B=B)(2)
    self.fc1 = get_hfta_op_for(nn.Linear, B=B)(9216, 128)
    self.fc2 = get_hfta_op_for(nn.Linear, B=B)(128, 10)
    self.dropout1 = get_hfta_op_for(nn.Dropout2d, B=B)(0.25)
    self.dropout2 = get_hfta_op_for(nn.Dropout2d, B=B)(0.5)

  # Minor modifications to the forward pass on special operators.
  # Check the documentation of each operator for details.
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.max_pool2d(x)
    x = self.dropout1(x)

    if self.B > 0:
      x = torch.flatten(x, 2)
      x = x.transpose(0, 1)
    else:
      x = torch.flatten(x, 1)

    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)

    if self.B > 0:
      output = F.log_softmax(x, dim=2)
    else:
      output = F.log_softmax(x, dim=1)

    return output

def train(config, model, device, train_loader, optimizer, epoch, B):
  """
  config: a dict defined by users to control the experiment
          See section: "Train the model"
  model: class Net defined in the code block above
  device: torch.device
  train_loader: torch.utils.data.dataloader.DataLoader
  optimizer: torch.optim
  epoch: int
  B: int, the number of models to be fused. When B == 0, we train the original 
     model as it is without enabling HFTA.
  """
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # Need to duplicate a single batch of input images into multiple batches to 
    # feed into the fused model.
    if B > 0:
      N = target.size(0)
      data = data.unsqueeze(1).expand(-1, B, -1, -1, -1)
      target = target.repeat(B)

    optimizer.zero_grad()
    output = model(data)

    # Also need to modify the loss function to take consideration on the fused 
    # model.
    # In the case:
    #   1) the loss function is reduced by averaging along the batch dimension.
    #   2) multiple models are horizontally fused via HFTA.
    # To make sure the mathematically equivalent gradients are derived by 
    # ".backward()", we need to scale the loss value by B.
    # You might refer to our paper for why such scaling is needed.
    if B > 0:
      loss = B * F.nll_loss(output.view(B * N, -1), target)
    else:
      loss = F.nll_loss(output, target)

    loss.backward()
    optimizer.step()
    if batch_idx % config["log_interval"] == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      if config["dry_run"]:
        break


def test(model, device, test_loader, B):
  """
  model: class Net defined in the code block above
  device: torch.device
  test_loader: torch.utils.data.dataloader.DataLoader
  B: int, the number of models to be fused. When B == 0, we test the original 
     model as it is without enabling HFTA.
  """
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      N = target.size(0)

      # Need to duplicate a single batch of input images into multiple batches 
      # to feed into the fused model.
      if B > 0:
        data = data.unsqueeze(1).expand(-1, B, -1, -1, -1)
        target = target.repeat(B)

      output = model(data)

      # Change the shape of the output to align with the loss function.
      if B > 0:
        output = output.view(B * N, -1)

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
  """
  config: a dict defined by users to control the experiment
  """
  random.seed(config["seed"])
  np.random.seed(config["seed"])
  torch.manual_seed(config["seed"])

  device = torch.device(config["device"])

  kwargs = {'batch_size': config["batch_size"]}
  kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))])

  # Determine the number of models that are horizontally fused together from the 
  # number of provided learning rates that need to be tested.
  B = len(config["lr"]) if config["use_hfta"] else 0

  dataset1 = datasets.MNIST('./data',
                            train=True,
                            download=True,
                            transform=transform)
  dataset2 = datasets.MNIST('./data', train=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

  # Specify the number of models that need to be fused horizontally together (B)
  # and create the fused model.
  model = Net(B).to(device)

  print('B={} lr={}'.format(B, config["lr"]), file=sys.stderr)

  # Convert the default optimizor (PyTorch Adadelta) to its HFTA version with 
  # get_hfta_optim_for(<default>, B).
  optimizer = get_hfta_optim_for(optim.Adadelta, B=B)(
      model.parameters(),
      lr=config["lr"] if B > 0 else config["lr"][0],
  )

  start = time.perf_counter()
  for epoch in range(1, config["epochs"] + 1):
    now = time.perf_counter()
    train(config, model, device, train_loader, optimizer, epoch, B)
    print('Epoch {} took {} s!'.format(epoch, time.perf_counter() - now))
  end = time.perf_counter()

  test(model, device, test_loader, B)

  print('All jobs Finished, Each epoch took {} s on average!'.format(
      (end - start) / (max(B, 1) * config["epochs"])))

# Enable HFTA, but not fusing models
# Only 1 model is trained
config = {
    "use_hfta": True,
    "device": "cuda",
    "batch_size": 64,
    "lr": [0.1],
    "gamma": 0.7,
    "epochs": 4,
    "seed": 1,
    "log_interval": 500,
    "dry_run": False,
    "save_model": False,
}

# # Enable HFTA and fuse 6 models
# config = {
#     "use_hfta": True,
#     "device": "cuda",
#     "batch_size": 64,
#     "lr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#     "gamma": 0.7,
#     "epochs": 4,
#     "seed": 1,
#     "log_interval": 500,
#     "dry_run": False,
#     "save_model": False,
# }


print(config)
main(config)
