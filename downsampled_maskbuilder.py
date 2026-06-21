import sys
import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import kornia.contrib as morph
import kornia.morphology as morph_kor
from kornia.filters import gaussian_blur2d

sys.path.append(os.path.join(os.path.dirname(__file__), 'PIDNet/'))
from models.pidnet import PIDNet

#from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class EdgeEnhanceConv(nn.Module):
    def __init__(self, device, dtype=torch.float16):
        super().__init__()
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=dtype, device=device)
        self.weight = nn.Parameter(kernel.view(1, 1, 3, 3), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x.half(), self.weight, padding=1)

class MaskRefiner(nn.Module):
    def __init__(self, device, dtype=torch.float16):
        super().__init__()
        self.edge_conv = EdgeEnhanceConv(device, dtype=dtype)
        self.kernel = torch.ones(21, 21, device=device, dtype=dtype)
        self.small_kernel = torch.ones(5, 5, device=device, dtype=dtype)

    def forward(self, building_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        building_mask = building_mask.half()
        edge_mask = edge_mask.half()

        edge_enh = self.edge_conv(edge_mask)
        edge_enh = torch.clamp(edge_enh, min=0)

        scale = 0.25
        small = F.interpolate(building_mask, scale_factor=scale, mode="nearest").half()
        closed_small = morph_kor.closing(small, self.kernel)
        building_closed = F.interpolate(closed_small, size=building_mask.shape[-2:], mode="nearest").half()

        for _ in range(2):
            building_closed = F.max_pool2d(building_closed, kernel_size=3, stride=1, padding=1).half()

        refined = building_closed * (1 + 0.2 * edge_enh)
        refined = torch.clamp(refined, 0, 1)

        return refined


class SimplifiedEdgeDetector:
    def __init__(self, device):

        self.device = device
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    def detect_edges_fast(self, image):

        kernel_size = 3 
        image = F.avg_pool2d(image, kernel_size, stride=1, padding=kernel_size//2)
        
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        direction = torch.atan2(grad_y, grad_x)
        
        threshold = magnitude.mean() + 0.5 * magnitude.std()
        edges = (magnitude > threshold).float() * 255
        
        return edges, direction

class MaskBuilder:
    def __init__(self, seg_dir: str, class_id: int = 2, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.building_id = class_id

        #self.model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128) #cityscape small
        #self.model = PIDNet(m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128) #cityscape medium
        self.model = PIDNet(m=2, n=3, num_classes=11,planes=32, ppm_planes=96, head_planes=128) #camvid_small
        #self.model = PIDNet(m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128) #camvid_medium

        # self.feature_extractor = SegformerImageProcessor.from_pretrained(seg_dir, local_files_only=True, do_reduce_labels = False)
        # self.model = SegformerForSemanticSegmentation.from_pretrained(seg_dir, local_files_only = True).eval().to(self.device)

        pretrained_dict = torch.load(seg_dir, map_location='cpu')
        if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        def strip_prefix(k: str):
            for p in ('module.', 'model.'):
                if k.startswith(p):
                    return k[len(p):]
            return k

        pretrained_dict = {strip_prefix(k): v for k, v in pretrained_dict.items()}
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=True)

        self.model.to(self.device).eval()

        self.edge_detector = SimplifiedEdgeDetector(device=self.device)
        self.dist_max = None
        #self.seg_scale_factor = 0.75
        #self.seg_scale_factor = 0.5
        #self.seg_scale_factor = 1.0
        self.seg_scale_factor = 0.25

        self.refiner = MaskRefiner(device=self.device)

    def _pad_to_multiple(self, x: Tensor, mult: int = 32) -> Tuple[Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        pad_h = (mult - (H % mult)) % mult
        pad_w = (mult - (W % mult)) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        return x_pad, (pad_h, pad_w)

    def _normalize_imagenet(self, x: Tensor) -> Tensor:  # 추후 일반화 개선 필요
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def segmask_builder(self, img_tensor: torch.Tensor, edge_mask):
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        _, _, H, W = img_tensor.shape

        proc_h, proc_w = int(H * self.seg_scale_factor), int(W * self.seg_scale_factor)


        x = img_tensor.to(self.device).float()
        x = F.interpolate(img_tensor, size=(proc_h, proc_w), 
                         mode='bilinear', align_corners=False).to(self.device).float()
        x = self._normalize_imagenet(x)
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, mult=32)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_pad)

        if isinstance(out, (list, tuple)):
            logits_pad = out[0]

        elif isinstance(out, dict):
            for k in ('main', 'out', 'pred', 'logits'):
                if k in out:
                    logits_pad = out[k]
                    break
            else:
                logits_pad = next(iter(out.values()))
        else:
            logits_pad = out

        if not torch.is_tensor(logits_pad):
            raise TypeError(f"PIDNet forward returned unsupported type: {type(out)}")

        # logits = F.interpolate(logits_pad, size=(H, W), mode='bilinear', align_corners=False)
        # pred = logits.argmax(dim=1, keepdim=True)
        # building_mask = (pred == self.building_id).float()

        # probs_pad = torch.softmax(logits_pad, dim=1)  # [1,C,hp,wp]
        # probs = F.interpolate(probs_pad, size=(H, W), mode='bilinear', align_corners=False)  # [1,C,H,W]
        # bld_prob = probs[:, self.building_id:self.building_id+1, :, :]                       # [1,1,H,W]
        # building_mask = (bld_prob >= 0.3).float()  

        logits = F.interpolate(logits_pad, size=(H, W), mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1, keepdim=True)  # [B,1,H,W]
        building_seg = (pred == self.building_id).float()
        building_seg = building_seg * (edge_mask > 0.01).float()
        building_mask = self.refiner(building_seg, edge_mask)
        #building_mask = building_seg * (1+0.5*edge_mask)
        return building_mask

    # def segmask_builder(self, img_tensor: torch.Tensor, edge_mask):

    #     if img_tensor.dim() == 3:
    #         img_tensor = img_tensor.unsqueeze(0)  # [1,3,H,W]

    #     img_tensor = img_tensor.to(self.device).float() / 255.0
    #     mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
    #     std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
    #     x = (img_tensor - mean) / std

    #     with torch.no_grad():
    #         logits = self.model(pixel_values=x).logits  # [B,num_classes,h,w]

    #     logits = F.interpolate(logits, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
    #     pred = logits.argmax(dim=1, keepdim=True)  
    #     building_seg = (pred == self.building_id).float()
    #     building_mask = building_seg * (1+0.5*edge_mask)
    #     return building_mask



    #     # building_logits = logits_pad[:, self.building_id:self.building_id+1, :, :]
    #     # building_logits = F.interpolate(building_logits, size=(H, W), mode="bilinear", align_corners=False)
    #     # building_mask = (building_logits > 0).float() 

    #     #print(building_mask.unique())
    #     # import matplotlib.pyplot as plt

    #     # plt.figure(figsize=(8,6))
    #     # plt.imshow(building_mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    #     # plt.colorbar(label="Building Probability")
    #     # plt.title("Building Probability Map (Continuous)")
    #     # plt.axis("off")
    #     # plt.show()

    #     return building_mask

    # def _normalize_imagenet(self, x: Tensor) -> Tensor:
    #     x = x.repeat(1, 3, 1, 1)
    #     return x/255.0

    # def segmask_builder(self, img_tensor: torch.Tensor) -> torch.Tensor:
    #     if img_tensor.dim() == 3:
    #         img_tensor = img_tensor.unsqueeze(0)
    #     _, _, H, W = img_tensor.shape

    #     proc_h, proc_w = int(H * self.seg_scale_factor), int(W * self.seg_scale_factor)


    #     x = img_tensor.to(self.device).float()
    #     x = F.interpolate(img_tensor, size=(proc_h, proc_w), 
    #                      mode='bilinear', align_corners=False).to(self.device).float()
    #     x = self._normalize_imagenet(x)
    #     x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, mult=32)

    #     self.model.eval()
    #     with torch.no_grad():
    #         out = self.model(x_pad)

    #     if isinstance(out, (list, tuple)):
    #         logits_pad = out[0]

    #     elif isinstance(out, dict):
    #         for k in ('main', 'out', 'pred', 'logits'):
    #             if k in out:
    #                 logits_pad = out[k]
    #                 break
    #         else:
    #             logits_pad = next(iter(out.values()))
    #     else:
    #         logits_pad = out

    #     if not torch.is_tensor(logits_pad):
    #         raise TypeError(f"PIDNet forward returned unsupported type: {type(out)}")

    #     # 1) 로짓 또는 확률을 '원본 해상도'로 보간
    #     #   (A) 로짓 보간 후 argmax
    #     # logits = F.interpolate(logits_pad, size=(H, W), mode='bilinear', align_corners=False)
    #     # pred = logits.argmax(dim=1, keepdim=True)
    #     # building_mask = (pred == self.building_id).float()

    #     # probs_pad = torch.softmax(logits_pad, dim=1)  # [1,C,hp,wp]
    #     # probs = F.interpolate(probs_pad, size=(H, W), mode='bilinear', align_corners=False)  # [1,C,H,W]
    #     # bld_prob = probs[:, self.building_id:self.building_id+1, :, :]                       # [1,1,H,W]
    #     # building_mask = (bld_prob >= 0.3).float()  # 필요 시 0.45~0.6로 튜닝

    #     logits = F.interpolate(logits_pad, size=(H, W), mode='bilinear', align_corners=False)
    #     pred = logits.argmax(dim=1, keepdim=True)  # [B,1,H,W]
    #     building_mask = (pred == self.building_id).float()


    #     # building_logits = logits_pad[:, self.building_id:self.building_id+1, :, :]
    #     # building_logits = F.interpolate(building_logits, size=(H, W), mode="bilinear", align_corners=False)
    #     # building_mask = (building_logits > 0).float() 

    #     #print(building_mask.unique())
    #     # import matplotlib.pyplot as plt

    #     # plt.figure(figsize=(8,6))
    #     # plt.imshow(building_mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    #     # plt.colorbar(label="Building Probability")
    #     # plt.title("Building Probability Map (Continuous)")
    #     # plt.axis("off")
    #     # plt.show()

    #     return building_mask

    def edgemask_builder(self, img_tensor: Tensor) -> Tuple[Tensor, Tensor]:

        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device).float()

        gray = img_tensor.mean(dim=1, keepdim=True)  # [1,1,H,W]
        edge_map, direction_map = self.edge_detector.detect_edges_fast(gray)  # edge_map: [1,1,H,W] or squeezed

        if edge_map.dim() == 2:
            edge_map = edge_map.unsqueeze(0).unsqueeze(0)
        elif edge_map.dim() == 3:
            edge_map = edge_map.unsqueeze(1)
        edge_map = edge_map.to(self.device).float() / 255.0

        dx_unit = torch.cos(direction_map)
        dy_unit = torch.sin(direction_map)
        grad_x = dx_unit * edge_map
        grad_y = dy_unit * edge_map
        return edge_map, grad_x, grad_y


    def distancemap_builder(self, edge_mask: Tensor, building_mask: Tensor) -> Tensor:
  
        if edge_mask.dim() == 2:
            edge_mask = edge_mask.unsqueeze(0).unsqueeze(0)
        elif edge_mask.dim() == 3:
            edge_mask = edge_mask.unsqueeze(1)

        edge_mask_bool = (edge_mask > 0)          # [1,1,H,W]
        building_mask_bool = (building_mask > 0)  # [1,1,H,W]

        region_mask = torch.logical_and(edge_mask_bool, building_mask_bool)
        in_building = torch.logical_xor(building_mask_bool, region_mask).float()

        dist = morph.distance_transform(region_mask.float())
        dist = dist * in_building  # 건물 내부에서만 거리 유지

        dist_flat = dist.flatten(2)  # [1,1,H*W]
        q95 = torch.quantile(dist_flat, 0.95, dim=2, keepdim=True)  # [1,1,1]
        self.dist_max = q95.unsqueeze(-1).clamp_min(1e-6)           # [1,1,1,1]
        dist = dist / (self.dist_max + 1e-6)

        dist_nl = torch.pow(dist, 2)
        dist_blur = gaussian_blur2d(dist_nl, kernel_size=(5, 5), sigma=(3.0, 3.0))
        
        #return dist
        return dist_blur

    def _odd(self, k: int) -> int:
        return k if k % 2 == 1 else k + 1

    # def pointmap_builder(self, building_mask: Tensor, edge_mask: Tensor, distance_map: Tensor,
    #                      max_val: float = 1.0, beta: float = 0.15) -> Tensor:
    #     edge_bool = edge_mask.bool()          # [1,1,H,W]
    #     building_bool = building_mask.bool()  # [1,1,H,W]
    #     intersection = torch.logical_and(edge_bool, building_bool)
    #     soft_region = torch.logical_xor(building_mask, intersection)

    #     rate = (distance_map * max_val).clamp_min(0.0)
    #     seg_dist = soft_region.float() * distance_map
    #     dist_poisson = seg_dist * rate

    #     seed = gaussian_blur2d(dist_poisson.float(), kernel_size=(3, 3), sigma=(1.5, 1.5))
    #     return seed

    def pointmap_builder(self, building_mask, edge_mask):

        edge_bool = edge_mask.bool()          # [1,1,H,W]
        building_bool = building_mask.bool()  # [1,1,H,W]
        intersection = torch.logical_and(edge_bool, building_bool)
        soft_region = torch.logical_xor(building_mask, intersection).float()
        seed = gaussian_blur2d(soft_region, kernel_size=(5, 5), sigma=(2.0, 2.0))
        return seed

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


    def generate_mask(self, img_tensor: Tensor, mask_tensor: Tensor, target_size: Tuple[int, int]) :

        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  
        _, _, H, W = img_tensor.shape

        img_tensor =  F.interpolate(img_tensor, size=(H, W), mode='bilinear', align_corners=False).to(self.device, non_blocking=True)

        img_tensor = img_tensor.to(self.device)
        edge_mask, grad_x, grad_y = self.edgemask_builder(img_tensor)  # [1,1,H,W] 둘 다
        building_mask = self.segmask_builder(img_tensor, edge_mask)

        building_mask = F.interpolate(building_mask, size=target_size, mode="nearest")
        edge_mask = F.interpolate(edge_mask, size=target_size, mode="nearest")
        grad_x = F.interpolate(grad_x, size=target_size, mode="bilinear")
        grad_y = F.interpolate(grad_y, size=target_size, mode="bilinear")

        valid_mask = F.interpolate(mask_tensor.float(), size=target_size, mode='nearest').to(self.device, non_blocking=True)

        building_mask = building_mask * valid_mask
        edge_mask = edge_mask * valid_mask

        distance_map = self.distancemap_builder(edge_mask, building_mask)  # [1,1,H,W]
        point_map = self.pointmap_builder(building_mask, edge_mask)

        distance_map = distance_map * valid_mask
        point_map = point_map * valid_mask
        grad_x = grad_x * valid_mask
        grad_y = grad_y * valid_mask

        w1 = torch.logical_and(edge_mask.bool(), building_mask.bool()).float()
        w5 = ((edge_mask.bool() & ~building_mask.bool()).float() * 0.5)
        w7 = self.normalize((point_map + distance_map)) * 0.7
        final_mask = torch.clamp(w1 + w5 + w7, 0, 1)

        return final_mask, edge_mask, distance_map, point_map, grad_x, grad_y, building_mask