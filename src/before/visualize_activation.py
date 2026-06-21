import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

class ActivationVisualizer:
    def __init__(self, save_dir="vis_outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.iteration = 0


    def visualize_mask_components(self, 
                              original_image,
                              building_mask, 
                              edge_mask,
                              distance_map,
                              point_map,
                              final_mask,
                              save_name=None):
        """마스크의 각 구성 요소를 시각화 (모두 grayscale)"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 원본 이미지
        if original_image.dim() == 4:
            img = original_image[0].cpu()
            if img.shape[0] == 1:  # grayscale
                img = img.squeeze(0)
            else:  # RGB
                img = img.permute(1, 2, 0)
        axes[0,0].imshow(img, cmap='gray' if img.dim()==2 else None)
        axes[0,0].set_title('Original Image')
        
        # 건물 마스크
        axes[0,1].imshow(building_mask.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title('Building Mask')
        
        # 엣지 마스크  
        axes[0,2].imshow(edge_mask.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title('Edge Mask')
        
        # 거리 맵
        axes[1,0].imshow(distance_map.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1,0].set_title('Distance Map')
        
        # 포인트 맵
        axes[1,1].imshow(point_map.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1,1].set_title('Point Map')
        
        # 최종 마스크
        im = axes[1,2].imshow(final_mask.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1,2].set_title('Final Mask (Sampling Weights)')
        plt.colorbar(im, ax=axes[1,2])
        
        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_masks.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def visualize_film_activation(self,
                                  features,
                                  context_map,
                                  gamma_maps,
                                  beta_maps,
                                  gate_values,
                                  save_name=None):
        """FiLM 활성화 패턴 시각화"""
        
        n_blocks = len(gamma_maps)
        fig, axes = plt.subplots(3, n_blocks+2, figsize=(4*(n_blocks+2), 12))
        
        # Feature map 평균 (첫 열)
        feat_mean = features.mean(dim=1).cpu()  # [B,H,W]
        axes[0,0].imshow(feat_mean[0], cmap='gray')  # 첫 번째 배치만
        axes[0,0].set_title('Feature Mean')
        
        # Context map 평균 (두 번째 열)  
        ctx_mean = context_map.mean(dim=1).cpu()  # [B,H,W]
        axes[0,1].imshow(ctx_mean[0], cmap='coolwarm')
        axes[0,1].set_title('Context Mean')
                
        # 각 블록별 gamma/beta
        for i, (gamma, beta) in enumerate(zip(gamma_maps, beta_maps)):
            col = i + 2
            
            # Gamma 효과 (곱셈)
            gamma_effect = (gamma - 1.0).abs().mean(dim=1).squeeze().cpu()
            im1 = axes[0, col].imshow(gamma_effect, cmap='RdBu_r', 
                                      vmin=-0.1, vmax=0.1)
            axes[0, col].set_title(f'Block {i} γ effect\n(gate={gate_values[i]:.3f})')
            
            # Beta 효과 (덧셈)
            beta_effect = beta.abs().mean(dim=1).squeeze().cpu()
            im2 = axes[1, col].imshow(beta_effect, cmap='viridis',
                                      vmin=0, vmax=0.1)
            axes[1, col].set_title(f'Block {i} β magnitude')
            
            # 결합 효과 (gamma와 beta의 상대적 영향)
            combined = gamma_effect + beta_effect
            im3 = axes[2, col].imshow(combined, cmap='hot')
            axes[2, col].set_title(f'Combined effect')
            
        # 빈 서브플롯 제거
        for i in range(3):
            for j in range(n_blocks+2):
                if j < 2 and i > 0:
                    axes[i,j].axis('off')
                else:
                    axes[i,j].axis('off')
                    
        plt.suptitle('FiLM Activation Patterns', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_film.png", dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()

    # def visualize_film_activation(self, original_images, features, context_map, gamma_maps, beta_maps, gate_values, save_name=None):

    #     # 배치 크기
    #     B = features.shape[0]

    #     # Feature / Context 평균 맵
    #     feat_mean = features.mean(dim=1).cpu()   # [B,H,W]
    #     ctx_mean  = context_map.mean(dim=1).cpu()  # [B,H,W]

    #     fig, axes = plt.subplots(B, 3, figsize=(12, 4*B))

    #     for b in range(B):
    #         # 원본 이미지
    #         img = original_images[b].permute(1,2,0).cpu().numpy()
    #         img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1]

    #         # FiLM combined 효과 (gamma+beta)
    #         combined_effect = 0
    #         for g, bmap in zip(gamma_maps, beta_maps):
    #             ge = (g[b] - 1.0).abs().mean(dim=0).cpu()
    #             be = bmap[b].abs().mean(dim=0).cpu()
    #             combined_effect += ge + be
    #         combined_effect = (combined_effect - combined_effect.min()) / (combined_effect.max() - combined_effect.min())

    #         # 시각화 1: 원본
    #         axes[b,0].imshow(img)
    #         axes[b,0].set_title(f'Original (sample {b})')
    #         axes[b,0].axis('off')

    #         # 시각화 2: Feature mean overlay
    #         axes[b,1].imshow(img)
    #         axes[b,1].imshow(feat_mean[b], cmap='gray', alpha=0.5)
    #         axes[b,1].set_title('Feature Mean overlay')
    #         axes[b,1].axis('off')

    #         # 시각화 3: FiLM 효과 overlay
    #         axes[b,2].imshow(img)
    #         axes[b,2].imshow(combined_effect, cmap='jet', alpha=0.5)
    #         axes[b,2].set_title('FiLM Combined Effect overlay')
    #         axes[b,2].axis('off')

    #     plt.suptitle('FiLM Activation Visualization (batch)', fontsize=16)
    #     plt.tight_layout()

    #     if save_name:
    #         plt.savefig(self.save_dir / f"{save_name}_film_overlay.png", dpi=150, bbox_inches='tight')
    #     plt.close()
        
    def visualize_sampling_distribution(self,
                                       pixel_positions,
                                       sampling_weights,
                                       sampled_indices,
                                       save_name=None):
        """샘플링 분포 시각화"""
        
        H, W = 32, 16  # ACE 기본 해상도
        
        # 샘플링 가중치 맵
        weight_map = sampling_weights.reshape(H, W).cpu()
        
        # 실제 샘플링된 위치
        sampled_map = torch.zeros(H, W)
        for idx in sampled_indices:
            h = idx // W
            w = idx % W
            sampled_map[h, w] += 1
        sampled_map = sampled_map / sampled_map.max() if sampled_map.max() > 0 else sampled_map
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 샘플링 가중치
        im1 = axes[0].imshow(weight_map, cmap='jet', vmin=0)
        axes[0].set_title('Sampling Weights (from mask)')
        plt.colorbar(im1, ax=axes[0])
        
        # 실제 샘플링 분포
        im2 = axes[1].imshow(sampled_map, cmap='hot')
        axes[1].set_title('Actual Sampling Distribution')
        plt.colorbar(im2, ax=axes[1])
        
        # 차이
        diff = (sampled_map - weight_map).abs()
        im3 = axes[2].imshow(diff, cmap='RdBu_r')
        axes[2].set_title('Difference')
        plt.colorbar(im3, ax=axes[2])
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_sampling.png", dpi=150)
        
        #plt.show()
        plt.close()