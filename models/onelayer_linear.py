"""Basic linear model with single hidden layer and optional nonlinearity."""

import torch.nn as nn
import torch.nn.functional as F

# TODO(loganesian): flexibility in projection head (i.e., full MLP)?
class OneLayerLinear(nn.Module):
  def __init__(self, input_size, num_ftrs, out_dim, activation=F.relu):
    super(OneLayerLinear, self).__init__()
    self.model = nn.Sequential()

    self.flatten = nn.Flatten()
    self.encoder = nn.Linear(input_size, num_ftrs, bias=True) # encoder
    self.activation = activation
    self.l1 = nn.Linear(num_ftrs, out_dim, bias=True) # projection

  def forward(self, x):
    h = self.encoder(self.flatten(x))
    if self.activation is not None: h = self.activation(h)
    z = self.l1(h)
    return h, z
