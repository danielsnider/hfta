def train(config, model, device, train_loader, optimizer, epoch):
  """
  config: a dict defined by users to control the experiment
          See section: "Train the model"
  model: class Net defined in the code block above
  device: torch.device
  train_loader: torch.utils.data.dataloader.DataLoader
  optimizer: torch.optim
  epoch: int
  """
  model.train() # not sure
  for batch_idx, (data, target) in enumerate(train_loader): # get batch of training samples (data) and labels (target)
    # data.shape: [batch, B, channel, pixel_x, pixel_y] <—each represents a pixel
    # target.shape: [batch] <—each represents a label
    data, target = data.to(device), target.to(device) # Convert from tf to cuda format.
    optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer. It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
    output = model(data) # forward pass of model, giving a logit for each class for each sample in the batch. Shape: [batch, class] <–each represents a logit.
    loss = F.nll_loss(output, target) # Compute difference between actual label and predicted logit
    loss.backward() # Computes gradient dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x.
    optimizer.step() # Use gradients and an optimizer algorithm to update the model parameters. Updates the value of x using the gradient x.grad. For example, the SGD optimizer performs: x += -lr * x.grad
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
