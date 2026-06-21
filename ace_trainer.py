# Copyright © Niantic, Inc. 2022.

import functools
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

import torch.nn.functional as F

from downsampled_maskbuilder import MaskBuilder
from pidicontext_adder import ContextMapBuilder
from itertools import chain

from visualize_activation import ActivationVisualizer

_logger = logging.getLogger(__name__)


def cuda_timeit(func):
    """Timing decorator for (possibly GPU) functions.

    CUDA kernels are launched asynchronously, so reading the wall clock right
    after a call measures only the *launch* time, not the actual GPU work. We
    therefore synchronize the device before and after the call so the elapsed
    time reflects the real compute cost.

    The decorator is stateful: it accumulates the elapsed time across calls and
    exposes it on the wrapper itself, so callers can read the timing without the
    measurement logic being interleaved with the training code.
        wrapper.last   -> elapsed time of the most recent call (seconds)
        wrapper.total  -> summed elapsed time over all calls (seconds)
        wrapper.calls  -> number of calls
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        wrapper.last = elapsed
        wrapper.total += elapsed
        wrapper.calls += 1
        return result

    wrapper.last = 0.0
    wrapper.total = 0.0
    wrapper.calls = 0
    wrapper.reset = lambda: wrapper.__dict__.update(last=0.0, total=0.0, calls=0)
    return wrapper


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass(frozen=True)
class RunConfig:
    """Immutable experiment configuration.

    Holds every hyperparameter that defines a training run. It is *frozen* so
    that, once training starts, the configuration cannot be mutated -- a run is
    therefore fully described by (and reproducible from) a single RunConfig
    object. Using an explicit dataclass instead of a raw argparse Namespace also
    makes the set of configurable knobs an explicit, type-annotated contract.
    """
    # Paths.
    scene: Path
    encoder_path: Path
    output_map_file: Path
    # Network / map.
    num_head_blocks: int
    use_homogeneous: bool
    # Optimization.
    learning_rate_min: float
    learning_rate_max: float
    training_buffer_size: int
    samples_per_image: int
    batch_size: int
    epochs: int
    use_half: bool
    # Reprojection loss.
    repro_loss_hard_clamp: int
    repro_loss_soft_clamp: int
    repro_loss_soft_clamp_min: int
    repro_loss_type: str
    repro_loss_schedule: str
    # Depth regularization.
    depth_min: float
    depth_target: float
    depth_max: float
    # Augmentation.
    use_aug: bool
    aug_rotation: int
    aug_scale: float
    image_resolution: int
    # Clustering (ensemble experiments; disabled by default).
    num_clusters: Optional[int]
    cluster_idx: Optional[int]
    # Visualization.
    render_visualization: bool
    render_target_path: Path
    render_flipped_portrait: bool
    render_map_error_threshold: int
    render_map_depth_filter: int
    render_camera_z_offset: int
    # Debug.
    debug_context: bool = False

    @classmethod
    def from_options(cls, options):
        """Build an immutable RunConfig from the argparse Namespace."""
        return cls(
            scene=options.scene,
            encoder_path=options.encoder_path,
            output_map_file=options.output_map_file,
            num_head_blocks=options.num_head_blocks,
            use_homogeneous=options.use_homogeneous,
            learning_rate_min=options.learning_rate_min,
            learning_rate_max=options.learning_rate_max,
            training_buffer_size=options.training_buffer_size,
            samples_per_image=options.samples_per_image,
            batch_size=options.batch_size,
            epochs=options.epochs,
            use_half=options.use_half,
            repro_loss_hard_clamp=options.repro_loss_hard_clamp,
            repro_loss_soft_clamp=options.repro_loss_soft_clamp,
            repro_loss_soft_clamp_min=options.repro_loss_soft_clamp_min,
            repro_loss_type=options.repro_loss_type,
            repro_loss_schedule=options.repro_loss_schedule,
            depth_min=options.depth_min,
            depth_target=options.depth_target,
            depth_max=options.depth_max,
            use_aug=options.use_aug,
            aug_rotation=options.aug_rotation,
            aug_scale=options.aug_scale,
            image_resolution=options.image_resolution,
            num_clusters=options.num_clusters,
            cluster_idx=options.cluster_idx,
            render_visualization=options.render_visualization,
            render_target_path=options.render_target_path,
            render_flipped_portrait=options.render_flipped_portrait,
            render_map_error_threshold=options.render_map_error_threshold,
            render_map_depth_filter=options.render_map_depth_filter,
            render_camera_z_offset=options.render_camera_z_offset,
            debug_context=getattr(options, "debug_context", False),
        )


@dataclass
class TrainerState:
    """Mutable runtime state that evolves during training.

    Kept separate from the immutable RunConfig so that "what was configured" and
    "what is happening right now" are not tangled together in the same object.
    """
    iteration: int = 0
    epoch: int = 0
    training_start: Optional[float] = None
    training_buffer: Optional[dict] = None


class TrainerACE:
    def __init__(self, options):
        # Separate the immutable experiment configuration (RunConfig) from the
        # mutable runtime state (TrainerState).
        self.config = RunConfig.from_options(options)
        self.state = TrainerState()

        self.device = torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.state.iteration = 0
        self.state.training_start = None
        self.num_data_loader_workers = 12

        pidnet_weights_path = Path(__file__).parent / "PIDNet" / "pretrained_models" / "camvid" / "PIDNet_S_Camvid_Test.pt"
        self.mask_builder = MaskBuilder(seg_dir=str(pidnet_weights_path), class_id=1, device=self.device)
        self.contextmap_builder = ContextMapBuilder(channel_in=5, channel_out=64).to(self.device).float()

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.config.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.config.use_half,
            image_height=self.config.image_resolution,
            augment=self.config.use_aug,
            aug_rotation=self.config.aug_rotation,
            aug_scale_max=self.config.aug_scale,
            aug_scale_min=1 / self.config.aug_scale,
            num_clusters=self.config.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.config.cluster_idx,    # Optional clustering for Cambridge experiments.
        )

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.config.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.config.encoder_path, map_location="cpu")
        
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.config.num_head_blocks,
            use_homogeneous=self.config.use_homogeneous
        )
        _logger.info(f"Loaded pretrained encoder from: {self.config.encoder_path}")

        _logger.info("Freezing encoder weights...")
        for param in self.regressor.encoder.parameters():
            param.requires_grad = False

        self.regressor = self.regressor.to(self.device)
        self.regressor.encoder.eval()
        self.regressor.heads.train()

        # Setup optimization parameters.
        #self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.config.learning_rate_min)
        
        _logger.info("Setting up optimizer for head and context modules...")
        trainable_params = chain(
            self.contextmap_builder.parameters(),
            self.regressor.heads.parameters()
        )
        
        self.optimizer = optim.AdamW(trainable_params, lr=self.config.learning_rate_min, weight_decay=1e-4)

        # Setup learning rate scheduler.
        steps_per_epoch = self.config.training_buffer_size // self.config.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.config.learning_rate_max,
                                                       epochs=self.config.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.config.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.config.epochs * self.config.training_buffer_size // self.config.batch_size
        self.iterations_output = 100 # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.config.repro_loss_soft_clamp,
            soft_clamp_min=self.config.repro_loss_soft_clamp_min,
            type=self.config.repro_loss_type,
            circle_schedule=(self.config.repro_loss_schedule == 'circle')
        )

        # Will be filled at the beginning of the training process.
        self.state.training_buffer = None

        # Generate video of training process
        if self.config.render_visualization:
            # infer rendering folder from map file name
            target_path = vutil.get_rendering_target_path(
                self.config.render_target_path,
                self.config.output_map_file)
            self.ace_visualizer = ACEVisualizer(
                target_path,
                self.config.render_flipped_portrait,
                self.config.render_map_depth_filter,
                mapping_vis_error_threshold=self.config.render_map_error_threshold)
        else:
            self.ace_visualizer = None

        self.debug_context = self.config.debug_context
        #self.debug_every = getattr(self.options, "debug_every", self.iterations_output)

        #self.vis_activations = ActivationVisualizer("./visualization")
        self.vis_counter = 0
        self.vis_frequency = 500

        def _summarize_channel_var(x_chw, topk=3):
            # x_chw : [C,H,W] or [B,C,H,W] -> 채널 분산 요약용
            if x_chw.dim() == 4:
                x_chw = x_chw.mean(0)  # [C,H,W]
            C = x_chw.shape[0]
            var = x_chw.view(C, -1).var(dim=1, unbiased=False)
            vals, idx = torch.sort(var, descending=True)
            top = [(int(idx[i]), float(vals[i])) for i in range(min(topk, C))]
            bot = [(int(idx[i]), float(vals[i])) for i in range(max(C-topk, 0), C)]
            return top, bot
        self._summarize_channel_var = _summarize_channel_var

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        if self.ace_visualizer is not None:

            # Setup the ACE render pipeline.
            self.ace_visualizer.setup_mapping_visualisation(
                self.dataset.pose_files,
                self.dataset.rgb_files,
                self.iterations // self.iterations_output + 1,
                self.config.render_camera_z_offset
            )

        # Reset the timing accumulators carried by the @cuda_timeit wrappers.
        type(self).create_training_buffer.reset()
        type(self).run_epoch.reset()

        self.state.training_start = time.time()

        # Create training buffer. Timing is handled by the @cuda_timeit decorator.
        self.create_training_buffer()
        creating_buffer_time = type(self).create_training_buffer.last
        _logger.info(f"Filled training buffer in {creating_buffer_time:.1f}s.")

        # Train the regression head. Each run_epoch call is timed by the decorator,
        # which accumulates the total across epochs.
        for self.state.epoch in range(self.config.epochs):

            progress = self.state.epoch / (self.config.epochs - 1)
            self.regressor.heads.film_gate_progress.fill_(progress)  #self.regressor.heads.film_gate_progress에서 film_gat_progress는 register buffer에 들어간 내용이긴 함

            self.run_epoch()

        training_time = type(self).run_epoch.total

        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.state.training_start:.1f} seconds.')

        if self.ace_visualizer is not None:

            # Finalize the rendering by animating the fully trained map.
            vis_dataset = CamLocDataset(
                root_dir=self.config.scene / "train",
                mode=0,
                use_half=self.config.use_half,
                image_height=self.config.image_resolution,
                augment=False) # No data augmentation when visualizing the map

            vis_dataset_loader = torch.utils.data.DataLoader(
                vis_dataset,
                shuffle=False, # Process data in order for a growing effect later when rendering
                num_workers=self.num_data_loader_workers)

            self.ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)

    @cuda_timeit
    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=8,
                                             drop_last=False)

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(dataset=self.dataset,
                                         sampler=batch_sampler,
                                         batch_size=None,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         timeout=60 if self.num_data_loader_workers > 0 else 0,
                                         )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.state.training_buffer = {
            'features': torch.empty((self.config.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.config.use_half], device=self.device),
            'context_map': torch.empty((self.config.training_buffer_size, 64),
                                   dtype=(torch.float32, torch.float16)[self.config.use_half], device=self.device),
            'target_px': torch.empty((self.config.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.config.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.config.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.config.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device)
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # === benchmark instrumentation: isolate peak GPU memory of the buffer-fill region ===
        # The ~6GB training buffer is already allocated above, so we snapshot the current
        # allocation as a baseline and reset the peak counter. The "intermediate" value we
        # report after filling is the high-water mark created *during* filling (the
        # per-iteration tensors), which is exactly what the eager->lazy change affects.
        _mem_base = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            _mem_base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        # ====================================================================================

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.config.training_buffer_size:
                dataset_passes += 1
                for image_B1HW, image_mask_B1HW, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in training_dataloader:
                    
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.config.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    B, C, H, W = features_BCHW.shape

                    # prob_mask, edge_map, distance_map, point_map, grad_x, grad_y, building_mask = self.mask_builder.generate_mask(image_B1HW, [H,W])

                    prob_mask, edge_map, distance_map, point_map, grad_x, grad_y, building_mask = self.mask_builder.generate_mask(image_B1HW, image_mask_B1HW, [H,W])

                    #prob_mask = continuous_mask # [1, 1, H, W]
                    #prob_mask = TF.resize(prob_mask, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias = False).to(self.device)  
                        
                    if self.debug_context and (buffer_idx % (10 * self.config.samples_per_image) == 0):
                        pm = prob_mask.float()
                        nz = float((pm > 0).float().mean())
                        mean_pm, std_pm = float(pm.mean()), float(pm.std())
                        hi = float((pm > 0.5).float().mean())
                        _logger.info(f"[DBG:mask] prob_mask: mean={mean_pm:.4f} std={std_pm:.4f} "
                                    f"nonzero={nz*100:.1f}% gt0.5={hi*100:.1f}%")

                    if prob_mask.sum().item() ==0:
                        continue



                    # edge_map = F.interpolate(edge_map, size=[H, W], mode="bilinear", align_corners=False).to(self.device)
                    # distance_map = F.interpolate(distance_map, size=[H, W], mode="bilinear", align_corners=False).to(self.device)
                    # point_map = F.interpolate(point_map, size=[H, W], mode="bilinear", align_corners=False).to(self.device)
                    # grad_x = F.interpolate(grad_x, size=[H, W], mode="bilinear", align_corners=False).to(self.device)
                    # grad_y = F.interpolate(grad_y, size=[H, W], mode="bilinear", align_corners=False).to(self.device)

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone() 
                    pixel_positions_B2HW = pixel_positions_B2HW[None] 
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  

                    # NOTE: poses/intrinsics are NOT expanded to the full (N, ...) shape
                    # here anymore. They are broadcast (one value per image), so we defer
                    # gathering them until after sampling (see below) and index only the
                    # selected rows. This avoids materializing large throwaway intermediates.

                    context_map_BCHW = self.contextmap_builder(edge_map, distance_map, point_map, grad_x, grad_y)
                    context_map_BCHW = F.interpolate(context_map_BCHW, size=(H, W), mode="bilinear", align_corners=False)
                    
                    
                    if self.debug_context and (buffer_idx % (10 * self.config.samples_per_image) == 0):
                        d_mean, d_std = float(distance_map.mean()), float(distance_map.std())
                        p_mean, p_std = float(point_map.mean()), float(point_map.std())
                        gx_mean, gx_std = float(grad_x.mean()), float(grad_x.std())
                        gy_mean, gy_std = float(grad_y.mean()), float(grad_y.std())

                        _logger.info(f"[DBG:ctx-in] dist={d_mean:.4f}±{d_std:.4f} | "
                                    f"point={p_mean:.4f}±{p_std:.4f} | "
                                    f"gx={gx_mean:.4f}±{gx_std:.4f} | gy={gy_mean:.4f}±{gy_std:.4f}")

                        top, bot = self._summarize_channel_var(context_map_BCHW.squeeze(0))
                        _logger.info(f"[DBG:ctx-pre] var_top={top} | var_bot={bot}")
                    
                    
                    
                    mean = torch.mean(context_map_BCHW, dim=(0, 2, 3), keepdim=True)
                    std = torch.std(context_map_BCHW, dim=(0, 2, 3), keepdim=True)
                    epsilon = 1e-6
                    context_map_BCHW = (context_map_BCHW - mean) / (std + epsilon)

                    def normalize_shape(tensor_in):
                        # BxCxHxW -> NxC
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    # Turn image mask into sampling weights (all equal).
                    #image_mask_B1HW = image_mask_B1HW.float()
                    #image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    prob_mask_N1 = normalize_shape(prob_mask).squeeze(1)
                    prob_mask_N1.clamp_(min=0.0)
                    if prob_mask_N1.sum() <= 0:
                        print("Warning: prob_mask_N1 sum=0 at buffer_idx", buffer_idx)
                        continue

                    features_to_select = min(self.config.samples_per_image * B, self.config.training_buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        prob_mask_N1.view(-1), features_to_select, replacement=True, generator=self.sampling_generator
                    )

                    # if buffer_idx % (50 * self.config.samples_per_image) == 0:  # 가끔씩만
                    #     self.vis_activations.visualize_mask_components(
                    #     image_B1HW[0:1],  
                    #     building_mask[0:1],
                    #     edge_map[0:1], 
                    #     distance_map[0:1],
                    #     point_map[0:1],
                    #     prob_mask[0:1],  
                    #     save_name=f"buffer_{buffer_idx}"
                    # )

                    # Decompose the flat sample indices back into (batch, row, col).
                    # normalize_shape maps position (b, h, w) -> flat index n = b*(H*W) + h*W + w,
                    # so we invert that mapping to gather directly from the BxCxHxW tensors.
                    HW = H * W
                    b_idx = sample_idxs // HW
                    rem = sample_idxs % HW
                    h_idx = rem // W
                    w_idx = rem % W

                    # Gather ONLY the sampled rows. Advanced indexing produces (S, C)
                    # directly, so the full (N, C) / (N, 3, 4) intermediates are never built.
                    batch_data = {
                        'features':       features_BCHW[b_idx, :, h_idx, w_idx],
                        'context_map':    context_map_BCHW[b_idx, :, h_idx, w_idx],
                        'target_px':      pixel_positions_B2HW[b_idx, :, h_idx, w_idx],
                        'gt_poses_inv':   gt_pose_inv_B44[b_idx, :3],
                        'intrinsics':     intrinsics_B33[b_idx],
                        'intrinsics_inv': intrinsics_inv_B33[b_idx],
                    }

                    # === benchmark instrumentation: per-iteration intermediate size (once) ===
                    # In the lazy version batch_data holds only the S sampled rows.
                    if buffer_idx == 0:
                        _bd = sum(v.element_size() * v.nelement() for v in batch_data.values())
                        _logger.info(f"[MEM:batch_data] per-iter intermediate = {_bd / 1024**2:.1f}MB "
                                     f"({batch_data['features'].shape[0]} rows)")
                    # =======================================================================

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.state.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    if buffer_idx >= self.config.training_buffer_size:
                        break

        # === benchmark instrumentation: report peak GPU memory of the buffer-fill region ===
        # Compare the "intermediate" value (peak above baseline) between before/after, NOT the
        # absolute peak, which is dominated by the persistent ~6GB buffer.
        if _mem_base is not None:
            torch.cuda.synchronize()
            _mem_peak = torch.cuda.max_memory_allocated()
            _logger.info(f"[MEM:buffer_fill] baseline={_mem_base / 1024**3:.3f}GB  "
                         f"peak={_mem_peak / 1024**3:.3f}GB  "
                         f"intermediate={(_mem_peak - _mem_base) / 1024**2:.1f}MB")
        # ===================================================================================

        buffer_memory = sum(v.element_size() * v.nelement() for v in self.state.training_buffer.values())
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    @cuda_timeit
    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(self.config.training_buffer_size, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, self.config.training_buffer_size, self.config.batch_size):
            batch_end = batch_start + self.config.batch_size

            # Drop last batch if not full.
            if batch_end > self.config.training_buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.state.training_buffer['features'][random_batch_indices].contiguous(),
                self.state.training_buffer['context_map'][random_batch_indices].contiguous(),
                self.state.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.state.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.state.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.state.training_buffer['intrinsics_inv'][random_batch_indices].contiguous()
            )
            self.state.iteration += 1

    def training_step(self, features_bC, context_map_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]
        context_channels = context_map_bC.shape[1]

        features_bCHW = features_bC.reshape(-1, 16, 32, channels).permute(0, 3, 1, 2)
        context_map_bCHW = context_map_bC.reshape(-1, 16, 32, context_channels).permute(0, 3, 1, 2)


        with autocast(enabled=self.config.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW, context_map_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        pred_px_b31[:, 2].clamp_(min=self.config.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        # Compute masks used to ignore invalid pixels.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.config.depth_min
        invalid_repro_b1 = reprojection_error_b1 > self.config.repro_loss_hard_clamp
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.config.depth_max

        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        # NaN 값이 마스크 계산에 사용될 경우, 마스크도 확인해볼 수 있습니다.
        # check_tensor(invalid_mask_b1.float(), "Debug. invalid_mask_b1")
        valid_mask_b1 = ~invalid_mask_b1

        if self.debug_context:
            # heads.debug_stats 는 ace_network.py 패치에서 채워짐
            ds = getattr(self.regressor.heads, "debug_stats", None)
            if isinstance(ds, dict):
                gate_sigmas = ds.get("gate_sigmas", None)
                gamma_stats = ds.get("gamma_stats", None)  # list of (mean,std) per block
                beta_stats  = ds.get("beta_stats", None)
                if gate_sigmas is not None:
                    _logger.info(f"[DBG:film] gate_sigma={['%.3f'%g for g in gate_sigmas]}")
                if gamma_stats is not None and beta_stats is not None:
                    # 각 블록별 (mean,std) 간단 요약
                    gs = [f"{m:.3f}±{s:.3f}" for (m, s) in gamma_stats]
                    bs = [f"{m:.3f}±{s:.3f}" for (m, s) in beta_stats]
                    _logger.info(f"[DBG:film] gamma_stats={gs}")
                    _logger.info(f"[DBG:film] beta_stats={bs}")

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.state.iteration)

        # Handle the invalid predictions
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.config.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        
        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()
        
        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # Optimization steps.
        old_optimizer_step = self.optimizer._step_count
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if hasattr(self.regressor.heads, "cross_align"):
            self.regressor.heads.cross_align.step()

        if self.state.iteration % self.iterations_output == 0:
            time_since_start = time.time() - self.state.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            _logger.info(f'Iteration: {self.state.iteration:6d} / Epoch {self.state.epoch:03d}|{self.config.epochs:03d}, '
                         f'Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s')

            if self.ace_visualizer is not None:
                vis_scene_coords = pred_scene_coords_b31.detach().cpu().squeeze().numpy()
                vis_errors = reprojection_error_b1.detach().cpu().squeeze().numpy()
                self.ace_visualizer.render_mapping_frame(vis_scene_coords, vis_errors)


        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()
        
        # if self.state.iteration % self.vis_frequency == 0:
        #     ds = getattr(self.regressor.heads, "debug_stats", None)
        #     if isinstance(ds, dict):
        #         features = features_bCHW.detach().cpu()
        #         context  = context_map_bCHW.detach().cpu()

        #         gamma_maps = [g.detach().cpu() for g in ds.get("gamma_raw", [])]
        #         beta_maps  = [b.detach().cpu() for b in ds.get("beta_raw", [])]
        #         gate_values = ds.get("gate_sigmas", [])

        #         if hasattr(self, "vis_activations"):
        #             self.vis_activations.visualize_film_activation(
        #                 features=features,
        #                 context_map=context,
        #                 gamma_maps=gamma_maps,
        #                 beta_maps=beta_maps,
        #                 gate_values=gate_values,
        #                 save_name=f"iter_{self.state.iteration}"
        #             )

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.config.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, self.config.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.config.output_map_file}")