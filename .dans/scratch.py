params = list(model.parameters())
[p.shape for p in params]

# B = 6
[torch.Size([6, 32, 1, 3, 3]),
 torch.Size([6, 32]),
 torch.Size([6, 64, 32, 3, 3]),
 torch.Size([6, 64]),
 torch.Size([6, 9216, 128]),
 torch.Size([6, 1, 128]),
 torch.Size([6, 128, 10]),
 torch.Size([6, 1, 10])]

# B = 1
[torch.Size([1, 32, 1, 3, 3]),
 torch.Size([1, 32]),
 torch.Size([1, 64, 32, 3, 3]),
 torch.Size([1, 64]),
 torch.Size([1, 9216, 128]),
 torch.Size([1, 1, 128]),
 torch.Size([1, 128, 10]),
 torch.Size([1, 1, 10])]

# B = 0
[torch.Size([32, 1, 3, 3]),
 torch.Size([32]),
 torch.Size([64, 32, 3, 3]),
 torch.Size([64]),
 torch.Size([128, 9216]),
 torch.Size([128]),
 torch.Size([10, 128]),
 torch.Size([10])]


     Hin, Win = input.size(-2), input.size(-1)
    if len(input.shape) == 4:
      input = input.reshape(-1,Hin, Win)
      input = input.unsqueeze(0)
      # print('WARNING: added dim')
