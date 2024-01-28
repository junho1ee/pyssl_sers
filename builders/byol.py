# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import copy
from PIL import Image


__all__ = ['BYOL']


class BYOL(nn.Module):
  """ 
  BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
  Link: https://arxiv.org/abs/2006.07733
  Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol
  """
  def __init__(self, backbone, feature_size, projection_dim=128, hidden_dim=2048, tau=0.996,
         transformations=None,):
    super().__init__()
    self.projection_dim = projection_dim
    self.tau = tau # EMA update
    self.backbone = backbone
    self.projector = MLP(feature_size, hidden_dim=hidden_dim, out_dim=projection_dim)
    self.online_encoder =  self.encoder = nn.Sequential(self.backbone, self.projector)
    self.online_predictor = MLP(in_dim=projection_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
    self.target_encoder = copy.deepcopy(self.online_encoder) # target must be a deepcopy of online, since we will use the backbone trained by online
    self._init_target_encoder()
    self.augment1 = transformations
    self.augment2 = transformations
  
  def forward(self, x1, x2):
    z1_o, z2_o = self.online_encoder(x1), self.online_encoder(x2)
    p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
    with torch.no_grad():
      self._momentum_update_target_encoder()
      z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2) 
    loss =  mean_squared_error(p1_o, z2_t) / 2 + mean_squared_error(p2_o, z1_t) / 2     
    return loss
  
  def _init_target_encoder(self):
    for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
      param_t.data.copy_(param_o.data)
      param_t.requires_grad = False
      
  @torch.no_grad()
  def _momentum_update_target_encoder(self):
    for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
      param_t.data = self.tau * param_t.data  + (1. - self.tau) * param_o.data
          

def mean_squared_error(p, z):
  p = F.normalize(p, dim=1)
  z = F.normalize(z, dim=1)
  return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


class MLP(nn.Module):
  """ Projection Head and Prediction Head for BYOL """
  def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
    super().__init__()

    self.layer1 = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.BatchNorm1d(hidden_dim),
      nn.ReLU(inplace=True)
    )
    self.layer2 = nn.Sequential(
      nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x 
  
  
if __name__ == '__main__':
  import torchvision
  backbone = torchvision.models.resnet50(pretrained=False)
  feature_size = backbone.fc.in_features
  backbone.fc = torch.nn.Identity()
    
  model = BYOL(backbone, feature_size, tau=0.996)
  
  x = torch.rand(4, 3, 224, 224)
  with torch.no_grad():
    loss = model.forward(x)
    print(f'loss = {loss}')