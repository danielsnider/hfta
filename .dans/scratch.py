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


# tracing through conv.py: _conv_forward()
            In[0] torch.Size([64, 6, 1, 28, 28]) # [batch, B, n_in_channels, height, width]
            In[0] self.conv1 = Conv2d(1, 32, 3, 1) # (n_in_channels, n_out_channels, kernel_size, stride) — weight shape: [B, 1, 32, 3, 3], bias shape: [B, 32]
            In[0] self.conv2 = Conv2d(32, 64, 3, 1) # (n_in_channels, n_out_channels, kernel_size, stride) — weight shape: [B, 32, 64, 3, 3], bias shape: [B, 64]
  def _conv_forward(self, input, weight):
    Hin, Win = input.size(3), input.size(4)
            In [1]: input.shape
            Out[1]: torch.Size([64, 6, 32, 26, 26]) # [batch, B, n_in_channels, height, width]
                                                    # SUGGESTION: B should come first which is how it is normally positioned
            In [2]: weight.shape
            Out[2]: torch.Size([6, 64, 32, 3, 3]) # [B, n_in_channels, n_out_channels, kernel_size, kernel_size] 
            In [12]: self.bias.shape
            Out[12]: torch.Size([6, 64]) # [B, n_out_channels]
            In [6]: print(Hin, Win)
            Out[6]: 26 26
    input = input.view(-1, self.B * self.in_channels, Hin, Win)
            Out[9]: torch.Size([64, 192, 26, 26]) # [batch, B * n_in_channels, height, width]
                # Input channels (n_in_channels) are no longer seperated by B. So RGB input to different models would become RGBRGBRGB.
    weight = weight.view(self.B * self.out_channels, # 6 * 64 = 384
                         self.in_channels // self.groups, # floor(32 / 1) = 32
                         *self.kernel_size) # use each item in tuple as an argument. So: 3, 3
            Out[10]: torch.Size([384, 32, 3, 3]) # [B * n_out_channels, n_in_channels, kernel_size, kernel_size]
                # Weight filters are no longer seperated by n_out_channels. So for each input channel, all output channel kernels are concatenated
    bias = (self.bias.view(self.B * self.out_channels) # 6 * 64 = 384
            if self.bias is not None else self.bias)
            Out[11]: torch.Size([384]) # Biases from all models are concatenated

    if self.padding_mode != 'zeros':
      y = F.conv2d(
          F.pad(input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode), weight, bias, self.stride, _pair(0),
          self.dilation, self.groups * self.B)
    else:
      y = F.conv2d(input, # [64, 192, 26, 26] # [batch, B * n_in_channels, height, width]
                   weight, # [384, 32, 3, 3] # [B * n_out_channels, n_in_channels, kernel_size, kernel_size]
                   bias, # [384] # [B * n_out_channels]
                   self.stride, # (1, 1)
                   self.padding, # (0, 0)
                   self.dilation, # (1, 1)
                   self.groups * self.B) # 1 * 6
            Out[19]: torch.Size([64, 384, 24, 24]) # [batch, B * n_out_channels, n_kernel_steps, n_kernel_steps]
    Hout, Wout = y.size(2), y.size(3)
            Out[20]: 24 24
    return y.view(-1, self.B, self.out_channels, Hout, Wout)
            Out[27]: torch.Size([64, 6, 64, 24, 24]) # [batch, B, n_out_channels, n_kernel_steps, n_kernel_steps]

Summary:
- Input concatenate on B * n_in_channels
- Weight concatenate on B * n_out_channels
- Bias concatenate on B * n_out_channels
- Output concatenated B * n_out_channels, so split those into separate dimensions, then return that
