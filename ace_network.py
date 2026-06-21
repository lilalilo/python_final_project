# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # FiLM 마지막 Conv는 tanh 전용 → Xavier
        if hasattr(m, 'activation') and m.activation == 'tanh_last':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1) 
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x

class FiLMGenerator(nn.Module):
    def __init__(self, channel_po=64, channel_en=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel_po, 64, 3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channel_en * 2, 1)
        )
        self.gamma_scale = nn.Parameter(torch.tensor(0.1))  
        self.beta_scale  = nn.Parameter(torch.tensor(0.1))  

    def forward(self, ctx):
        raw = self.net(ctx)
        gamma_raw, beta_raw = torch.chunk(raw, 2, dim=1)
        gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
        beta  = self.beta_scale * torch.tanh(beta_raw)
        return gamma, beta
    
class ShiftCrossSimilarity(nn.Module):
    def __init__(self, channel_feat, channel_con, projection=64, radius=1, temperature_init=0.1, learnable_temp=True):
        super().__init__()
        self.radius = int(radius)
        self.kernel_size = 2 * radius + 1
        self.projection = projection

        # projection layers
        self.feature_proj = nn.Conv2d(channel_feat, projection, 1, bias=True)
        self.context_proj = nn.Conv2d(channel_con, projection, 1, bias=True)

        # learnable temperature
        if learnable_temp:
            self.log_t = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        else:
            self.register_buffer("log_t", torch.log(torch.tensor(temperature_init)))

        # offsets list (dx,dy)
        offsets = [(dx, dy) for dy in range(-radius, radius+1) for dx in range(-radius, radius+1)]
        self.register_buffer("dx", torch.tensor([o[0] for o in offsets], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("dy", torch.tensor([o[1] for o in offsets], dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, feature_map, context_map):
        B, _, H, W = feature_map.shape

        # 1) projection + normalize
        f_proj = F.normalize(self.feature_proj(feature_map), dim=1, eps=1e-6)   # [B,d,H,W]
        c_proj = F.normalize(self.context_proj(context_map), dim=1, eps=1e-6)   # [B,d,H,W]

        # 2) shift-based correlation
        sims = []
        for dy in range(-self.radius, self.radius+1):
            for dx in range(-self.radius, self.radius+1):
                shifted = torch.roll(c_proj, shifts=(dy, dx), dims=(2, 3))  # [B,d,H,W]
                sim = (f_proj * shifted).sum(dim=1, keepdim=True)           # [B,1,H,W]
                sims.append(sim)
        sims = torch.cat(sims, dim=1)  # [B,K,H,W], K=(2r+1)^2

        # 3) softmax over local offsets
        t = torch.exp(self.log_t).clamp(min=1e-4, max=10.0)
        attn = F.softmax(sims / t, dim=1)  # [B,K,H,W]

        # 4) expected dx, dy (soft-argmax)
        dx = (attn * self.dx.view(1,-1,1,1)).sum(dim=1, keepdim=True)
        dy = (attn * self.dy.view(1,-1,1,1)).sum(dim=1, keepdim=True)

        # 5) grid sample
        xs = torch.linspace(-1, 1, W, device=context_map.device).view(1,1,1,W).expand(B,1,H,W)
        ys = torch.linspace(-1, 1, H, device=context_map.device).view(1,1,H,1).expand(B,1,H,W)

        dx_n = 2.0 * dx / max(W-1, 1)
        dy_n = 2.0 * dy / max(H-1, 1)

        grid = torch.stack([ys.squeeze(1) + dy_n.squeeze(1),
                            xs.squeeze(1) + dx_n.squeeze(1)], dim=-1)

        aligned = F.grid_sample(context_map, grid, mode='bilinear',
                                padding_mode='border', align_corners=False)

        debug = {
            "temp": float(t.detach()),
            "dx_mean": float(dx.mean().detach()), "dy_mean": float(dy.mean().detach()),
        }
        return aligned, debug
    
class GridCrossSimilarityShift(nn.Module):
    def __init__(self, channel_feat, channel_con, projection=64, radius=1, temperature_init=0.5):
        super().__init__()
        self.radius = int(radius)
        self.kernel_size = 2 * self.radius + 1
        self.projection = projection

        # 1x1 conv projection
        self.feature_proj = nn.Conv2d(channel_feat, projection, 1, bias=True)
        self.context_proj = nn.Conv2d(channel_con, projection, 1, bias=True)

        # learnable temperature
        self.log_t = nn.Parameter(torch.log(torch.tensor(temperature_init)))

        # offset buffers (dx, dy for each local position)
        offset = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                offset.append((dx, dy))
        self.register_buffer("dx", torch.tensor([o[0] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("dy", torch.tensor([o[1] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, feature_map, context_map):
        B, _, H, W = feature_map.shape

        # 1) projection + normalize
        feat_proj = F.normalize(self.feature_proj(feature_map), dim=1, eps=1e-6)  # [B,d,H,W]
        ctx_proj = F.normalize(self.context_proj(context_map), dim=1, eps=1e-6)   # [B,d,H,W]

        # 2) base grid (normalized coordinates)
        xs = torch.linspace(-1.0, 1.0, W, device=context_map.device)
        ys = torch.linspace(-1.0, 1.0, H, device=context_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        base_grid = base_grid.unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]

        # 3) sample shifted context maps
        shifted_ctx = []
        for dx, dy in zip(self.dx.view(-1), self.dy.view(-1)):
            dx_n = 2.0 * dx.item() / max(W - 1, 1)
            dy_n = 2.0 * dy.item() / max(H - 1, 1)
            grid = base_grid.clone()
            grid[..., 0] = grid[..., 0] + dx_n  # x
            grid[..., 1] = grid[..., 1] + dy_n  # y
            ctx_shifted = F.grid_sample(ctx_proj, grid, mode="bilinear",
                                        padding_mode="border", align_corners=True)
            shifted_ctx.append(ctx_shifted)
        patches = torch.stack(shifted_ctx, dim=2)  # [B,d,K,H,W]

        # 4) similarity
        feat_exp = feat_proj.unsqueeze(2)          # [B,d,1,H,W]
        sim = (patches * feat_exp).sum(dim=1)      # [B,K,H,W]

        # 5) attention
        t = torch.exp(self.log_t).clamp(min=1e-4, max=10.0)
        attn = F.softmax(sim / t, dim=1)           # [B,K,H,W]

        # 6) expected offset
        dx = (attn * self.dx.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        dy = (attn * self.dy.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]

        dx_n = 2.0 * dx / max(W - 1, 1)
        dy_n = 2.0 * dy / max(H - 1, 1)
        grid = torch.stack([base_grid[..., 0] + dx_n.squeeze(1),
                            base_grid[..., 1] + dy_n.squeeze(1)], dim=-1)

        # 7) align context map
        aligned = F.grid_sample(context_map, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)

        debug = {
            "temp": float(t.detach()),
            "dx_mean": float(dx.mean().detach()),
            "dy_mean": float(dy.mean().detach())
        }

        return aligned, debug
    
# class CrossSimilarityShift(nn.Module):
#     def __init__(self, channel_feat, channel_con, projection = 64, radius = 1, temperature_init =0.1, learnable_temp=True, align_context_to_feature = True):
#         super().__init__()

#         self.radius = int(radius)
#         self.kernel_size = 2*self.radius + 1
#         self.projection = int(projection)
#         self.align_cf = align_context_to_feature

#         # 1*1 conv
#         self.feature_proj = nn.Conv2d(channel_feat, self.projection, 1, bias = True)
#         self.context_proj = nn.Conv2d(channel_con, self.projection, 1, bias = True)

#         if learnable_temp:
#             self.log_t = nn.Parameter(torch.log(torch.tensor(temperature_init)))
#         else:
#             self.register_buffer("log_t", torch.log(torch.tensor(temperature_init)))
        
#         offset = []

#         for dy in range(-self.radius, self.radius+1):
#             for dx in range(-self.radius, self.radius+1):
#                 offset.append((dx,dy))
#         self.register_buffer("dx", torch.tensor([o[0] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))    
#         self.register_buffer("dy", torch.tensor([o[1] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))

#     def _compute_orientation(self, fmap):
#         gray = fmap.mean(dim=1, keepdim=True)  # 채널 평균
#         sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], 
#                             dtype=fmap.dtype, device=fmap.device).view(1,1,3,3)
#         sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], 
#                             dtype=fmap.dtype, device=fmap.device).view(1,1,3,3)
#         grad_x = F.conv2d(gray, sobel_x, padding=1)
#         grad_y = F.conv2d(gray, sobel_y, padding=1)
#         return torch.atan2(grad_y, grad_x)
    
#     def forward(self, feature_map, context_map):
#         B, _, H, W = feature_map.shape

#         # 1) project & L2-normalize
#         feature_projection = self.feature_proj(feature_map)          # [B, d, H, W]
#         context_projection = self.context_proj(context_map)          # [B, d, H, W]
#         feature_projection = F.normalize(feature_projection, dim=1, eps=1e-6)
#         context_projection = F.normalize(context_projection, dim=1, eps=1e-6)

#         feature_orient = self._compute_orientation(feature_map)  # [B,1,H,W]
#         context_orient = self._compute_orientation(context_map)  # [B,1,H,W]
#         orient_weight = torch.cos(feature_orient - context_orient).clamp(min=0.0)
#         orient_weight = orient_weight.view(B, 1, H*W)

#         # 2) unfold local patches (from context)
#         patches = F.unfold(
#             context_projection, 
#             kernel_size=self.kernel_size, 
#             padding=self.radius, 
#             stride=1
#         )  # [B, d*ks*ks, HW]
#         Bp, DCk, L = patches.shape
#         K_actual = DCk // self.projection   # should be (2r+1)^2
#         if K_actual != self.kernel_size * self.kernel_size:
#             _logger.error(f"[ALIGN] K mismatch: unfold returned K={K_actual}, "
#                         f"while expected K={(self.kernel_size*self.kernel_size)} "
#                         f"(radius={self.radius}, ks={self.kernel_size})")
#             raise RuntimeError("Local window K mismatch – check kernel_size/padding/radius.")
#         K = K_actual
#         patches = patches.view(B, self.projection, K, H*W)           # [B, d, K, HW]

#         # 3) flatten feature to match
#         flat_feature = feature_projection.view(B, self.projection, H*W).unsqueeze(2)  # [B, d, 1, HW]

#         # 4) local similarity per offset -> [B, K, HW]
#         # scores = (patches * flat_feature).sum(dim=1)
#         scores = (patches * flat_feature).sum(dim=1)   # [B, K, HW]
#         scores = scores * orient_weight     

        

#         # 5) softmax with temperature
#         t = torch.exp(self.log_t).clamp(min=1e-4, max=10.0)
#         attn = F.softmax(scores / t, dim=1)                           # [B, K, HW]

#         # 6) reshape to [B,K,H,W] and compute soft-argmax shifts
#         attn_hw = attn.view(B, K, H, W)                               # [B, K, H, W]
#         # 안전성 체크: dx/dy K가 맞는지
#         if self.dx.shape[1] != K or self.dy.shape[1] != K:
#             _logger.error(f"[ALIGN] dx/dy buffer K mismatch: dxK={self.dx.shape[1]}, dyK={self.dy.shape[1]}, attnK={K}")
#             raise RuntimeError("dx/dy buffer size does not match K – check radius/offset construction.")
#         dx = (attn_hw * self.dx.view(1, K, 1, 1)).sum(dim=1)          # [B, H, W]
#         dy = (attn_hw * self.dy.view(1, K, 1, 1)).sum(dim=1)          # [B, H, W]
#         dx = dx.view(B, 1, H, W)
#         dy = dy.view(B, 1, H, W)

#         # 7) build sampling grid (note: grid_sample expects (y,x))
#         # normalized base grid in [-1,1]
#         xs = torch.linspace(-1.0, 1.0, W, device=context_map.device).view(1,1,1,W).expand(B,1,H,W)
#         ys = torch.linspace(-1.0, 1.0, H, device=context_map.device).view(1,1,H,1).expand(B,1,H,W)

#         dx_n = 2.0 * dx / max(W - 1, 1)
#         dy_n = 2.0 * dy / max(H - 1, 1)

#         grid_x = xs + dx_n   # [B,1,H,W]
#         grid_y = ys + dy_n   # [B,1,H,W]

#         # stack as (y,x) for grid_sample
#         grid = torch.stack([grid_y.squeeze(1), grid_x.squeeze(1)], dim=-1)  # [B,H,W,2], (y,x)

#         C_aligned = F.grid_sample(
#             context_map, grid, mode='bilinear',
#             padding_mode='border', align_corners=False
#         )

#         debug = {
#             "temp": float(t.detach()),
#             "dx_mean": float(dx.mean().detach()), "dx_std": float(dx.std().detach()),
#             "dy_mean": float(dy.mean().detach()), "dy_std": float(dy.std().detach()),
#             "attn_entropy": float((-(attn+1e-8).log() * attn).mean().detach()),
#             "K": int(K),
#         }

#         # _logger.info(
#         #     f"[ALIGN] K={debug['K']} t={debug['temp']:.3f} H={debug['attn_entropy']:.3f} "
#         #     f"dx={debug['dx_mean']:.2f}±{debug['dx_std']:.2f} dy={debug['dy_mean']:.2f}±{debug['dy_std']:.2f}"
#         # )

#         return C_aligned, debug

class CrossSimilarityShift(nn.Module):   
    def __init__(self, channel_feat, channel_con, projection=64, radius=1, temperature_init=0.1,
                 learnable_temp=True, align_context_to_feature=True, alpha_init=0.0):
        super().__init__() 
        self.radius = int(radius)
        self.kernel_size = 2 * self.radius + 1
        self.projection = int(projection)
        self.align_cf = align_context_to_feature

        # projection layers
        self.feature_proj = nn.Conv2d(channel_feat, self.projection, 1, bias=True)
        self.context_proj = nn.Conv2d(channel_con, self.projection, 1, bias=True)

        # temperature
        if learnable_temp:
            self.log_t = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        else:
            self.register_buffer("log_t", torch.log(torch.tensor(temperature_init)))

        # orientation weight factor (curriculum learning)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=False)

        # dx/dy offsets
        offset = [(dx, dy) for dy in range(-self.radius, self.radius+1)
                           for dx in range(-self.radius, self.radius+1)]
        self.register_buffer("dx", torch.tensor([o[0] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("dy", torch.tensor([o[1] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))

        # Sobel filters for orientation
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Cache for the normalized base sampling grid. The grid only depends on
        # (H, W, device); during training the patch size is fixed (16x32), so this
        # is otherwise recomputed identically on every forward. Keyed by (H, W, device).
        self._grid_cache = {}

    def _compute_orientation(self, fmap):
        # grayscale로 합치기 (채널 평균)
        gray = fmap.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.atan2(gy, gx)  # [-pi, pi]

    def forward(self, feature_map, context_map):
        B, _, H, W = feature_map.shape

        # project & normalize
        feat_proj = F.normalize(self.feature_proj(feature_map), dim=1, eps=1e-6)
        ctx_proj = F.normalize(self.context_proj(context_map), dim=1, eps=1e-6)

        # unfold patches from context
        patches = F.unfold(ctx_proj, kernel_size=self.kernel_size, padding=self.radius, stride=1)
        Bp, DCk, L = patches.shape
        K = DCk // self.projection
        patches = patches.view(B, self.projection, K, H*W)

        feat_flat = feat_proj.view(B, self.projection, H*W).unsqueeze(2)
        scores = (patches * feat_flat).sum(dim=1)  # [B,K,HW]

        # softmax with temperature
        t = torch.exp(self.log_t).clamp(min=1e-4, max=10.0)
        attn = F.softmax(scores / t, dim=1).view(B, K, H, W)

        # --- orientation weight 추가 ---
        orient_feat = self._compute_orientation(feature_map)  # [B,1,H,W]
        orient_ctx  = self._compute_orientation(context_map)  # [B,1,H,W]
        orient_diff = orient_feat - orient_ctx
        orient_w = torch.cos(orient_diff)                     # [-1,1]
        orient_w = (orient_w + 1) / 2                         # [0,1]
        attn = (1 - self.alpha) * attn + self.alpha * (attn * orient_w)

        # dx, dy expectation
        dx = (attn * self.dx.view(1, K, 1, 1)).sum(dim=1, keepdim=True)
        dy = (attn * self.dy.view(1, K, 1, 1)).sum(dim=1, keepdim=True)

        # grid sampling
        # Base grid depends only on (H, W, device): memoize it instead of rebuilding
        # the same linspace tensors on every forward call.
        key = (H, W, context_map.device)
        if key not in self._grid_cache:
            xs_base = torch.linspace(-1.0, 1.0, W, device=context_map.device).view(1, 1, 1, W)
            ys_base = torch.linspace(-1.0, 1.0, H, device=context_map.device).view(1, 1, H, 1)
            self._grid_cache[key] = (xs_base, ys_base)
        xs_base, ys_base = self._grid_cache[key]
        xs = xs_base.expand(B, 1, H, W)
        ys = ys_base.expand(B, 1, H, W)
        dx_n = 2.0 * dx / max(W-1,1)
        dy_n = 2.0 * dy / max(H-1,1)
        grid = torch.stack([ys.squeeze(1)+dy_n.squeeze(1), xs.squeeze(1)+dx_n.squeeze(1)], dim=-1)
        aligned = F.grid_sample(context_map, grid, mode="bilinear", padding_mode="border", align_corners=False)

        debug = {
            "temp": float(t.detach()),
            "dx_mean": float(dx.mean().detach()),
            "dy_mean": float(dy.mean().detach()),
            "alpha": float(self.alpha.detach())
        }
        return aligned, debug

    
class FastCrossSimilarityShift(nn.Module):
    def __init__(self, channel_feat, channel_con, projection=64, radius=1, temperature_init=0.1):
        super().__init__()
        self.radius = int(radius)
        self.kernel_size = 2 * self.radius + 1
        self.projection = projection

        # projection layers
        self.feature_proj = nn.Conv2d(channel_feat, projection, 1, bias=True)
        self.context_proj = nn.Conv2d(channel_con, projection, 1, bias=True)

        # learnable temperature
        self.log_t = nn.Parameter(torch.log(torch.tensor(temperature_init)))

        # offset buffer
        offset = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                offset.append((dx, dy))
        self.register_buffer("dx", torch.tensor([o[0] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("dy", torch.tensor([o[1] for o in offset], dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, feature_map, context_map):
        B, _, H, W = feature_map.shape

        # project + normalize
        feat_proj = F.normalize(self.feature_proj(feature_map), dim=1, eps=1e-6)
        ctx_proj = F.normalize(self.context_proj(context_map), dim=1, eps=1e-6)

        # build shifted contexts (roll-based instead of unfold)
        patches = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                shifted = torch.roll(ctx_proj, shifts=(dy, dx), dims=(2, 3))
                patches.append(shifted)
        patches = torch.stack(patches, dim=2)  # [B, d, K, H, W]

        # similarity: dot product
        feat_exp = feat_proj.unsqueeze(2)  # [B, d, 1, H, W]
        sim = (patches * feat_exp).sum(dim=1)  # [B, K, H, W]

        # attention with temperature
        t = torch.exp(self.log_t).clamp(min=1e-4, max=10.0)
        attn = F.softmax(sim / t, dim=1)  # [B, K, H, W]

        # expected shift (soft-argmax)
        dx = (attn * self.dx.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        dy = (attn * self.dy.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]

        # base grid
        xs = torch.linspace(-1.0, 1.0, W, device=context_map.device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1.0, 1.0, H, device=context_map.device).view(1, 1, H, 1).expand(B, 1, H, W)

        dx_n = 2.0 * dx / max(W - 1, 1)
        dy_n = 2.0 * dy / max(H - 1, 1)

        grid = torch.stack([ys.squeeze(1) + dy_n.squeeze(1),
                            xs.squeeze(1) + dx_n.squeeze(1)], dim=-1)  # [B,H,W,2]

        # align context map
        aligned = F.grid_sample(context_map, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)

        debug = {
            "temp": float(t.detach()),
            "dx_mean": float(dx.mean().detach()),
            "dy_mean": float(dy.mean().detach())
        }

        return aligned, debug


class FiLMedHead(nn.Module):
    """
    기존 Head의 각 Residual Block에 FiLM을 적용하여 컨텍스트 정보를 주입하는 새로운 헤드.
    """
    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(FiLMedHead, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels
        self.head_channels = 512
       
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.aligner = CrossSimilarityShift(
            channel_feat=self.head_channels,   # 512
            channel_con=64,                    # context map channels
            projection=64,
            radius=1,
            temperature_init=0.1,
            learnable_temp=True,
            align_context_to_feature=True
        )

        # self.aligner = GridCrossSimilarityShift(
        #     channel_feat=self.head_channels,   # 512
        #     channel_con=64,                    # context map channels
        #     projection=64,
        #     radius=1,
        #     temperature_init=0.3,
        #     #learnable_temp=True,
        # )

        initial_gate = -2.5
        final_gate = -1.5

        self.register_buffer("film_gate_initial", torch.tensor(initial_gate))
        self.register_buffer("film_gate_final", torch.tensor(final_gate))
        self.register_buffer("film_gate_progress", torch.tensor(0.0))

        self.film_generators = nn.ModuleList(
            [FiLMGenerator(channel_po=64, channel_en=self.head_channels) for _ in range(num_head_blocks + 1)]
        )

        # self.film_gates = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(-4.0)) for _ in range(num_head_blocks + 1)]
        # )

        # self.film_gates = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(-1.5)) for _ in range(num_head_blocks + 1)]
        # )

        self.film_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(-2.0)) for _ in range(num_head_blocks + 1)]
        )
        
        self.ctx_smoother = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.ctx_dropout_p = 0.2

        self.res_blocks = []
        
        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))
            super(FiLMedHead, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(FiLMedHead, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(FiLMedHead, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        
        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)

        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

        # ---- Debug containers (no effect on forward output) ----
        self.debug_stats = {
            "gate_sigmas": None,   # list[float] per FiLM site (0..num_blocks)
            "gamma_stats": None,   # list[(mean,std)] per FiLM site
            "beta_stats":  None,   # list[(mean,std)] per FiLM site
        }

    
    # def _gate(self, idx):
    #     # σ(-4)≈0.018 -> 거의 0에서 시작
    #     return torch.sigmoid(self.film_gates[idx])
    
    # def _gate(self, idx):
    #     # 선형 보간: progress=0 -> initial, progress=1 -> final
    #     gate_value = (1 - self.film_gate_progress) * self.film_gate_initial + \
    #                 self.film_gate_progress * self.film_gate_final
    #     return torch.sigmoid(gate_value)
    
    def _gate(self, idx):
        schedule = (1 - self.film_gate_progress) * self.film_gate_initial + \
                self.film_gate_progress * self.film_gate_final
        gate_value = self.film_gates[idx] + schedule
        return torch.sigmoid(gate_value)
    

    def _apply_film(self, res, context_map, gen_idx, gamma_scale=0.1, beta_scale=0.1):
        ctx = F.interpolate(context_map, size=res.shape[2:], mode='bilinear', align_corners=False)
        gamma_raw, beta_raw = self.film_generators[gen_idx](ctx)
        gamma = 1.0 + gamma_scale * torch.tanh(gamma_raw)  # ~ [0.9, 1.1]
        beta  = beta_scale * torch.tanh(beta_raw)          # ~ [-0.1, 0.1]
        res = gamma * res + beta
        res = torch.clamp(res, min=-1e6, max=1e6)
        return res
    
    def forward(self, res, context_map, return_debug=False):
        # 컨텍스트 저역통과 + (선택) 드롭아웃
        ctx = self.ctx_smoother(context_map)
        if self.training and self.ctx_dropout_p > 0:
            # 채널 단위 드롭 (Spatial은 유지)
            m = (torch.rand((ctx.shape[0], ctx.shape[1], 1, 1), device=ctx.device) > self.ctx_dropout_p).float()
            ctx = ctx * m

        ctx = F.interpolate(ctx, size=res.shape[2:], mode='bilinear', align_corners=False)

        ctx_aligned, align_dbg = self.aligner(res, ctx)

        #visualize film
        debug_gamma_maps = []
        debug_beta_maps = []
        debug_gate_values = []

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        # --- 첫 주입: x에 FiLM 적용 후 skip 연결 ---
        #ctx_resized = F.interpolate(ctx, size=res.shape[2:], mode='bilinear', align_corners=False)
        
        gamma, beta = self.film_generators[0](ctx_aligned)
        g = self._gate(0)
        x_film = gamma * x + beta
        x = (1 - g) * x + g * x_film

        gate_sigmas = []
        gamma_stats = []
        beta_stats  = []
        with torch.no_grad():
            gate_sigmas.append(float(g.detach()))
            gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
            beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))

        res = self.head_skip(res) + x

        for i, res_block in enumerate(self.res_blocks):
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            # per-block도 같은 ctx_aligned 사용 (해상도 동일)
            gamma, beta = self.film_generators[i+1](ctx_aligned)
            g = self._gate(i+1)
            x_film = gamma * x + beta
            x = (1 - g) * x + g * x_film

            with torch.no_grad():
                gate_sigmas.append(float(g.detach()))
                gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
                beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)
        if self.use_homogeneous:
            sc = sc[:, :3] / (1e-6 + F.softplus(sc[:, 3:4], beta=1.0))
        sc += self.mean

        # 디버그 저장 (기존 + align)
        self.debug_stats["gate_sigmas"] = gate_sigmas
        self.debug_stats["gamma_stats"] = gamma_stats
        self.debug_stats["beta_stats"]  = beta_stats
        self.debug_stats["align"] = align_dbg

        if return_debug:
            debug_info = {
                "features": res.detach().cpu(),
                "context": ctx_aligned.detach().cpu(),
                "gamma_maps": debug_gamma_maps,
                "beta_maps": debug_beta_maps,
                "gate_values": debug_gate_values,
                "align": align_dbg
            }
            return sc, debug_info
        else:
            return sc


# class FiLMedHead(nn.Module):
#     """
#     CrossSimilarityShift 없는 버전 - 성능 비교용
#     """
#     def __init__(self,
#                  mean,
#                  num_head_blocks,
#                  use_homogeneous,
#                  homogeneous_min_scale=0.01,
#                  homogeneous_max_scale=4.0,
#                  in_channels=512):
#         super(FiLMedHead, self).__init__()

#         self.use_homogeneous = use_homogeneous
#         self.in_channels = in_channels
#         self.head_channels = 512
       
#         self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

#         # CrossSimilarityShift 대신 간단한 1x1 conv projection
#         self.ctx_projector = nn.Conv2d(64, 64, 1, 1, 0)
#         self.align_gate = nn.Parameter(torch.tensor(0.5))  # 학습 가능한 mixing weight

#         self.film_generators = nn.ModuleList(
#             [FiLMGenerator(channel_po=64, channel_en=self.head_channels) for _ in range(num_head_blocks + 1)]
#         )

#         self.film_gates = nn.ParameterList(
#             [nn.Parameter(torch.tensor(-2.0)) for _ in range(num_head_blocks + 1)]
#         )
        
#         self.ctx_smoother = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
#         self.ctx_dropout_p = 0.2

#         self.res_blocks = []
        
#         for block in range(num_head_blocks):
#             self.res_blocks.append((
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#             ))
#             super(FiLMedHead, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
#             super(FiLMedHead, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
#             super(FiLMedHead, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

#         self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
#         self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        
#         if self.use_homogeneous:
#             self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)
#             self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
#             self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
#             self.register_buffer("max_inv_scale", 1. / self.max_scale)
#             self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
#             self.register_buffer("min_inv_scale", 1. / self.min_scale)
#         else:
#             self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)
            
#         self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

#         self.debug_stats = {
#             "gate_sigmas": None,
#             "gamma_stats": None,
#             "beta_stats":  None,
#         }
    
#     def _gate(self, idx):
#         return torch.sigmoid(self.film_gates[idx])
    
#     def forward(self, res, context_map):
#         # 컨텍스트 저역통과 + 드롭아웃
#         ctx = self.ctx_smoother(context_map)
#         if self.training and self.ctx_dropout_p > 0:
#             m = (torch.rand((ctx.shape[0], ctx.shape[1], 1, 1), device=ctx.device) > self.ctx_dropout_p).float()
#             ctx = ctx * m

#         ctx = F.interpolate(ctx, size=res.shape[2:], mode='bilinear', align_corners=False)

#         # CrossSimilarityShift 대신 간단한 projection + gating
#         ctx_projected = self.ctx_projector(ctx)
#         ctx_aligned = torch.sigmoid(self.align_gate) * ctx_projected + (1 - torch.sigmoid(self.align_gate)) * ctx

#         x = F.relu(self.res3_conv1(res))
#         x = F.relu(self.res3_conv2(x))
#         x = F.relu(self.res3_conv3(x))

#         # 첫 FiLM 적용
#         gamma, beta = self.film_generators[0](ctx_aligned)
#         g = self._gate(0)
#         x_film = gamma * x + beta
#         x = (1 - g) * x + g * x_film

#         gate_sigmas = []
#         gamma_stats = []
#         beta_stats  = []
#         with torch.no_grad():
#             gate_sigmas.append(float(g.detach()))
#             gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
#             beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))

#         res = self.head_skip(res) + x

#         # Residual blocks with FiLM
#         for i, res_block in enumerate(self.res_blocks):
#             x = F.relu(res_block[0](res))
#             x = F.relu(res_block[1](x))
#             x = F.relu(res_block[2](x))

#             gamma, beta = self.film_generators[i+1](ctx_aligned)
#             g = self._gate(i+1)
#             x_film = gamma * x + beta
#             x = (1 - g) * x + g * x_film

#             with torch.no_grad():
#                 gate_sigmas.append(float(g.detach()))
#                 gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
#                 beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))

#             res = res + x

#         # Final layers
#         sc = F.relu(self.fc1(res))
#         sc = F.relu(self.fc2(sc))
#         sc = self.fc3(sc)
#         if self.use_homogeneous:
#             sc = sc[:, :3] / (1e-6 + F.softplus(sc[:, 3:4], beta=1.0))
#         sc += self.mean

#         self.debug_stats["gate_sigmas"] = gate_sigmas
#         self.debug_stats["gamma_stats"] = gamma_stats
#         self.debug_stats["beta_stats"]  = beta_stats

#         return sc


# class FiLMedHead(nn.Module):
#     """
#     CrossSimilarityShift 없는 버전 - 성능 비교용
#     """
#     def __init__(self,
#                  mean,
#                  num_head_blocks,
#                  use_homogeneous,
#                  homogeneous_min_scale=0.01,
#                  homogeneous_max_scale=4.0,
#                  in_channels=512):
#         super(FiLMedHead, self).__init__()

#         self.use_homogeneous = use_homogeneous
#         self.in_channels = in_channels
#         self.head_channels = 512
       
#         self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
#         self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

#         # CrossSimilarityShift 대신 간단한 1x1 conv projection
#         self.ctx_projector = nn.Conv2d(64, 64, 1, 1, 0)
#         self.align_gate = nn.Parameter(torch.tensor(0.5))  # 학습 가능한 mixing weight

#         self.film_generators = nn.ModuleList(
#             [FiLMGenerator(channel_po=64, channel_en=self.head_channels) for _ in range(num_head_blocks + 1)]
#         )

#         self.film_gates = nn.ParameterList(
#             [nn.Parameter(torch.tensor(-2.0)) for _ in range(num_head_blocks + 1)]
#         )
        
#         self.ctx_smoother = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
#         self.ctx_dropout_p = 0.2

#         self.res_blocks = []
        
#         for block in range(num_head_blocks):
#             self.res_blocks.append((
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#                 nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
#             ))
#             super(FiLMedHead, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
#             super(FiLMedHead, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
#             super(FiLMedHead, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

#         self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
#         self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        
#         if self.use_homogeneous:
#             self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)
#             self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
#             self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
#             self.register_buffer("max_inv_scale", 1. / self.max_scale)
#             self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
#             self.register_buffer("min_inv_scale", 1. / self.min_scale)
#         else:
#             self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)
            
#         self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

#         self.debug_stats = {
#             "gate_sigmas": None,
#             "gamma_stats": None,
#             "beta_stats":  None,
#         }
    
#     def _gate(self, idx):
#         return torch.sigmoid(self.film_gates[idx])
    
    
#     def forward(self, res, context_map):
#         # 컨텍스트 저역통과 + 드롭아웃
#         ctx = self.ctx_smoother(context_map)
#         if self.training and self.ctx_dropout_p > 0:
#             m = (torch.rand((ctx.shape[0], ctx.shape[1], 1, 1), device=ctx.device) > self.ctx_dropout_p).float()
#             ctx = ctx * m

#         ctx = F.interpolate(ctx, size=res.shape[2:], mode='bilinear', align_corners=False)

#         # 간단 projection + gating
#         ctx_projected = self.ctx_projector(ctx)
#         ctx_aligned = torch.sigmoid(self.align_gate) * ctx_projected + (1 - torch.sigmoid(self.align_gate)) * ctx

#         x = F.relu(self.res3_conv1(res))
#         x = F.relu(self.res3_conv2(x))
#         x = F.relu(self.res3_conv3(x))

#         # ---- 첫 FiLM 적용 ----
#         gamma, beta = self.film_generators[0](ctx_aligned)
#         g = self._gate(0)
#         x_film = gamma * x + beta
#         x = (1 - g) * x + g * x_film

    
#         with torch.no_grad():
#             diff_map = (x_film - x).abs().mean(dim=1, keepdim=True)  # 채널 평균 차이
#             self.debug_stats["first_diff_map"] = diff_map.cpu()      # [B,1,H,W]
#             self.debug_stats["gamma_first"] = gamma[0].detach().cpu()  # 한 샘플만 저장
#             self.debug_stats["beta_first"]  = beta[0].detach().cpu()

#         gate_sigmas = []
#         gamma_stats = []
#         beta_stats  = []
#         with torch.no_grad():
#             gate_sigmas.append(float(g.detach()))
#             gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
#             beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))

#         res = self.head_skip(res) + x

#         # ---- Residual blocks with FiLM ----
#         for i, res_block in enumerate(self.res_blocks):
#             x = F.relu(res_block[0](res))
#             x = F.relu(res_block[1](x))
#             x = F.relu(res_block[2](x))

#             gamma, beta = self.film_generators[i+1](ctx_aligned)
#             g = self._gate(i+1)
#             x_film = gamma * x + beta
#             x = (1 - g) * x + g * x_film

#             with torch.no_grad():
#                 gate_sigmas.append(float(g.detach()))
#                 gamma_stats.append((float(gamma.mean().detach()), float(gamma.std().detach())))
#                 beta_stats.append((float(beta.mean().detach()), float(beta.std().detach())))
#                 # 각 블록별 diff 저장 가능
#                 diff_map = (x_film - x).abs().mean(dim=1, keepdim=True)
#                 self.debug_stats[f"diff_map_block{i+1}"] = diff_map.cpu()

#             res = res + x

#         # ---- Final layers ----
#         sc = F.relu(self.fc1(res))
#         sc = F.relu(self.fc2(sc))
#         sc = self.fc3(sc)
#         if self.use_homogeneous:
#             sc = sc[:, :3] / (1e-6 + F.softplus(sc[:, 3:4], beta=1.0))
#         sc += self.mean

#         # Debug 저장
#         self.debug_stats["gate_sigmas"] = gate_sigmas
#         self.debug_stats["gamma_stats"] = gamma_stats
#         self.debug_stats["beta_stats"]  = beta_stats

#         return sc



class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim)
        self.heads = FiLMedHead(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features, context_map):
        return self.heads(features, context_map)

    def forward(self, inputs, context_map):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features, context_map)