#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

# === 추가: segmentation + context ===
from downsampled_maskbuilder import MaskBuilder
from pidicontext_adder import ContextMapBuilder
# ===================================

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # DSACStar RANSAC parameters.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64)
    parser.add_argument('--threshold', '-t', type=float, default=10)
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100)
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100)

    # Visualization params
    parser.add_argument('--render_visualization', type=_strtobool, default=False)
    parser.add_argument('--render_target_path', type=Path, default='renderings')
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False)
    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False)
    parser.add_argument('--render_pose_error_threshold', type=int, default=20)
    parser.add_argument('--render_map_depth_filter', type=int, default=10)
    parser.add_argument('--render_camera_z_offset', type=int, default=4)
    parser.add_argument('--render_frame_skip', type=int, default=1)

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)

    # Dataset
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    testset_loader = DataLoader(testset, shuffle=False, num_workers=num_workers)

    # Encoder & Head weights
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # === PIDNet weights 경로 (이 워크스페이스 기준 상대 경로) ===
    _workspace_dir = Path(__file__).parent
    #pidnet_weights_path = str(_workspace_dir / "PIDNet" / "pretrained_models" / "cityscapes" / "PIDNet_S_Cityscapes_test.pt")
    #pidnet_weights_path = str(_workspace_dir / "PIDNet" / "pretrained_models" / "cityscapes" / "PIDNet_M_Cityscapes_test.pt")
    pidnet_weights_path = str(_workspace_dir / "PIDNet" / "pretrained_models" / "camvid" / "PIDNet_S_Camvid_Test.pt")
    mask_builder = MaskBuilder(
        seg_dir=pidnet_weights_path,
        class_id=1,
        device=device
    )

    # mask_builder = MaskBuilder(
    #     seg_dir=pidnet_weights_path,
    #     class_id=1,
    #     device=device
    # )

    contextmap_builder = ContextMapBuilder(channel_in=5, channel_out=64).to(device).float()
    # ====================================

    # Network
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = network.to(device)
    network.eval()

    # Logs
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    avg_batch_time = 0
    num_batches = 0
    rErrs, tErrs = [], []
    pct10_5 = pct5 = pct2 = pct1 = 0

    # Visualization
    if opt.render_visualization:
        target_path = vutil.get_rendering_target_path(opt.render_target_path, opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,
            image_height=opt.image_resolution,
        )
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=num_workers)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop
    with torch.no_grad():
        for image_B1HW, image_mask_B1HW, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()

            image_B1HW = image_B1HW.to(device, non_blocking=True)
            image_mask_B1HW = image_mask_B1HW.to(device, non_blocking=True)

            with autocast(enabled=True):
                features_BCHW = network.get_features(image_B1HW)
                B, C, H_feat, W_feat = features_BCHW.shape

                _, edge_map, distance_map, point_map, grad_x, grad_y, _ = \
                    mask_builder.generate_mask(image_B1HW, image_mask_B1HW, [H_feat, W_feat])

                context_map_BCHW = contextmap_builder(edge_map, distance_map, point_map, grad_x, grad_y)

                mean = torch.mean(context_map_BCHW, dim=(0, 2, 3), keepdim=True)
                std = torch.std(context_map_BCHW, dim=(0, 2, 3), keepdim=True)
                context_map_BCHW = (context_map_BCHW - mean) / (std + 1e-6)

                scene_coordinates_B3HW = network(image_B1HW, context_map_BCHW)

            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            for scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path in zip(
                scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames
            ):
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                frame_name = Path(frame_path).name
                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries)

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 0.1: pct10_5 += 1
                if r_err < 5 and t_err < 0.05: pct5 += 1
                if r_err < 2 and t_err < 0.02: pct2 += 1
                if r_err < 1 and t_err < 0.01: pct1 += 1

                out_pose = out_pose.inverse()
                t = out_pose[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / (angle + 1e-12)
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    avg_time = avg_batch_time / num_batches

    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")
    _logger.info('Accuracy:')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()