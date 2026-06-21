
import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedContextMapBuilder(nn.Module):
    def __init__(self, channel_in=5, channel_out=64):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        
        # Simplified architecture - removed intermediate steps
        self.conv = nn.Conv2d(channel_in, channel_out, 3, 1, 1, bias=True)
        
        # Pre-compute normalization buffers to avoid repeated calculations
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.momentum = 0.1
        self.eps = 1e-6

    def _efficient_normalize(self, x):
        """More efficient normalization using running statistics"""
        if self.training:
            # Update running statistics during training
            batch_mean = x.mean()
            batch_var = x.var()
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            return (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics during inference
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
    
    def forward(self, *maps):
        # Concatenate and normalize in one step
        x = torch.cat(maps, dim=1)
        x = self._efficient_normalize(x)
        
        # Single convolution instead of multiple layers
        x = self.conv(x)
        return x

class ContextMapBuilder(nn.Module):
    def __init__(self, channel_in=5, channel_out=64):
        super(ContextMapBuilder, self).__init__()
        
        self.channel_in = channel_in
        self.channel_out = channel_out

        # edge, gradx, grady  
        self.structure = nn.Sequential(  
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True),
        )

        # self.structure = nn.Sequential(  
        #     nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        # ) 


        # distance, point
        self.geometry = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        fusion_in = 16+24
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.channel_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self,edge_map, distance_map, point_map, grad_x, grad_y):

        # B, _, H, W = edge_map.shape
        # device = edge_map.device
        # xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
        # yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)

        #structure = torch.cat([edge_map, xx, yy, grad_x, grad_y], dim=1)
        
        structure = torch.cat([edge_map, grad_x, grad_y], dim=1)
        structure_feat = self.structure(structure)

        geometry = torch.cat([distance_map, point_map], dim=1)
        geometry_feat = self.geometry(geometry)

        fused = torch.cat([structure_feat, geometry_feat], dim=1)
        context_map = self.fusion(fused)

        return context_map
