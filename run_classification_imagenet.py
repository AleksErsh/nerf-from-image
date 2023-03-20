import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import sys
import math

from torch.utils import tensorboard
from torch import nn
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf
from typing import Tuple, List
from pathlib import Path

import arguments
from data import loaders
from data import imagenet_loader
from lib import pose_utils
from lib import nerf_utils
from lib import utils
from lib import fid
from lib import ops
from lib import metrics
from lib import pose_estimation
from lib import dino
from models import generator
from models import discriminator
from models import encoder

# Set global gpu settings
cfg_ = OmegaConf.load("config.yaml")
GPU_IDS = list(range(cfg_.gpus))

if cfg_.gpus > 0 and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def render(
    target_model,
    height,
    width,
    tform_cam2world,
    focal_length,
    center,
    bbox,
    model_input,
    cfg
):
    """
    Render 2D image using target model and camera pose parameters
    """
    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(),
            cfg.dataset_config.scene_range
        )

    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins,
        ray_directions,
        near_thresh,
        far_thresh,
        cfg.rendering.depth_samples_per_ray,
        randomize=cfg.rendering.randomize,
    )

    if cfg.rendering.force_no_cam_grad:
        query_points = query_points.detach()
        depth_values = depth_values.detach()
        ray_directions = ray_directions.detach()

    if cfg.use_viewdir:
        viewdirs = ray_directions.unsqueeze(-2)
    else:
        viewdirs = None

    model_outputs = target_model(
        viewdirs, model_input,
        ['sampler'] + cfg.rendering.extra_model_outputs,
        cfg.rendering.extra_model_inputs
    )
    radiance_field_sampler = model_outputs['sampler']
    del model_outputs['sampler']

    request_sampler_outputs = ['sigma', 'rgb']

    if cfg.rendering.compute_normals:
        assert cfg.use_sdf
        request_sampler_outputs.append('normals')

    if cfg.rendering.compute_semantics:
        assert cfg.generator.attention_values > 0
        request_sampler_outputs.append('semantics')

    if cfg.rendering.compute_coords:
        request_sampler_outputs.append('coords')

    sampler_outputs_coarse = radiance_field_sampler(query_points, request_sampler_outputs)
    sigma = sampler_outputs_coarse['sigma'].view(*query_points.shape[:-1], -1)
    rgb = sampler_outputs_coarse['rgb'].view(*query_points.shape[:-1], -1)
    normals = None
    semantics = None
    coords = None
    if cfg.rendering.compute_normals:
        normals = sampler_outputs_coarse['normals'].view(*query_points.shape[:-1], -1)
    if cfg.rendering.compute_semantics:
        semantics = sampler_outputs_coarse['semantics'].view(*query_points.shape[:-1], -1)
    if cfg.rendering.compute_coords:
        coords = sampler_outputs_coarse['coords'].view(*query_points.shape[:-1], -1)

    if cfg.fine_sampling:
        z_vals = depth_values
        with torch.no_grad():
            weights = nerf_utils.render_volume_density_weights_only(
                sigma.squeeze(-1), ray_origins, ray_directions,
                depth_values
            ).flatten(0, 2)

            # Smooth weights as in EG3D
            weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = F.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = nerf_utils.sample_pdf(
                z_vals_mid.flatten(0, 2),
                weights[..., 1:-1],
                cfg.rendering.depth_samples_per_ray,
                deterministic = not cfg.rendering.randomize,
            )
            z_samples = z_samples.view(*z_vals.shape[:3], z_samples.shape[-1])

        z_values_sorted, z_indices_sorted = torch.sort(
            torch.cat((z_vals, z_samples), dim=-1),
            dim=-1
        )
        query_points_fine = ray_origins[..., None, :] + \
            ray_directions[..., None, :] * z_samples[..., :, None]

        sampler_outputs_fine = radiance_field_sampler(
            query_points_fine,
            request_sampler_outputs
        )
        sigma_fine = sampler_outputs_fine['sigma'].view(
            *query_points_fine.shape[:-1], -1)
        rgb_fine = sampler_outputs_fine['rgb'].view(
            *query_points_fine.shape[:-1], -1)
        normals_fine = None
        semantics_fine = None
        coords_fine = None
        if cfg.rendering.compute_normals:
            normals_fine = sampler_outputs_fine['normals'].view(
                *query_points_fine.shape[:-1], -1)
        if cfg.rendering.compute_semantics:
            semantics_fine = sampler_outputs_fine['semantics'].view(
                *query_points_fine.shape[:-1], -1)
        if cfg.rendering.compute_coords:
            coords_fine = sampler_outputs_fine['coords'].view(
                *query_points_fine.shape[:-1], -1)

        sigma = torch.cat((sigma, sigma_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, sigma.shape[-1])
        )
        rgb = torch.cat((rgb, rgb_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, rgb.shape[-1])
        )
        if normals_fine is not None:
            normals = torch.cat((normals, normals_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, normals.shape[-1])
            )
        if semantics_fine is not None:
            semantics = torch.cat((semantics, semantics_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, semantics.shape[-1])
            )
        if coords_fine is not None:
            coords = torch.cat((coords, coords_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, coords.shape[-1])
            )
        depth_values = z_values_sorted

    if coords is not None:
        semantics = coords

    rgb_predicted, depth_predicted, mask_predicted, \
     normals_predicted, semantics_predicted = nerf_utils.render_volume_density(
        sigma.squeeze(-1),
        rgb,
        ray_origins,
        ray_directions,
        depth_values,
        normals,
        semantics,
        white_background=cfg.dataset_config.white_background
    )

    return rgb_predicted, depth_predicted, mask_predicted, \
            normals_predicted, semantics_predicted, model_outputs

class ParallelModel(nn.Module):

    def __init__(self, cfg, model=None, model_ema=None, lpips_net=None):
        super().__init__()
        self.cfg = cfg
        self.resolution = cfg.generator.resolution
        self.model = model
        self.model_ema = model_ema
        self.lpips_net = lpips_net
        self.use_ema = cfg.generator.use_ema
        self.pretrain_sdf = cfg.generator.pretrain_sdf
        self.encoder_output = cfg.generator.encoder_output
        self.res_multiplier = cfg.generator.res_multiplier
        self.ray_multiplier = cfg.generator.ray_multiplier

    def forward(self,
                tform_cam2world,
                focal,
                center,
                bbox,
                c,
                closure=None,
                closure_params=None):
        model_to_use = self.model_ema if self.use_ema else self.model
        if self.pretrain_sdf:
            return model_to_use(
                None,
                c,
                request_model_outputs=['sdf_distance_loss', 'sdf_eikonal_loss'])
        if self.encoder_output:
            return model_to_use.emb(c)

        output = render(
            model_to_use,
            int(self.resolution * self.res_multiplier),
            int(self.resolution * self.res_multiplier),
            tform_cam2world,
            focal,
            center,
            bbox,
            c,
            self.cfg
        )
        if closure is not None:
            return closure(
                self, output[0], output[2], output[4], output[-1],
                **closure_params)  # RGB, alpha, semantics, extra_outptus

        return output


def load_encoders(cfg) -> dict:
    """
    Load pretrained encoders for ImageNet classes
    specified in config file
    """
    encoders_checkpoints_dir = os.path.join(cfg.root_path, cfg.encoder_weights_dir)

    encoders = {}
    for cls_name, cls_ in cfg.ImageNet.items():
        cls_encoder = encoder.BootstrapEncoder(
            latent_dim=512,
            pretrained=False).to(DEVICE)

        cls_encoder = nn.DataParallel(cls_encoder, GPU_IDS)
        checkpoint_path = os.path.join(encoders_checkpoints_dir, cls_["encoder_checkpoint_path"])
        with utils.open_file(checkpoint_path, 'rb') as checkpoint:
            cls_encoder.load_state_dict(
                torch.load(checkpoint, map_location='cpu')['model_coord']
            )
        cls_encoder.requires_grad_(False)
        cls_encoder.eval()
        encoders[cls_name] = cls_encoder

    return encoders

def load_generators(cfg) -> Tuple[dict, dict]:
    """
    Load pretrained generators for ImageNet classes
    specified in config file
    """
    generators_checkpoints_dir = os.path.join(cfg.root_path, cfg.generator_weights_dir)

    model = None

    models_ema = {}
    parallel_models = {}
    for cls_name, cls in cfg.ImageNet.items():
        loss_fn_lpips = metrics.LPIPSLoss().to(DEVICE)

        model_ema = create_generator(cfg)
        model_ema.eval()
        model_ema.requires_grad_(False)

        parallel_model = nn.DataParallel(
            ParallelModel(
                cfg,
                model=model,
                model_ema=model_ema,
                lpips_net=loss_fn_lpips
            ),
            GPU_IDS
        ).to(DEVICE)

        total_params = 0
        for param in model_ema.parameters():
            total_params += param.numel()

        models_ema[cls_name] = model_ema
        parallel_models[cls_name] = parallel_model

        checkpoint_path = os.path.join(generators_checkpoints_dir, cls["generator_checkpoint_path"])
        with utils.open_file(checkpoint_path, 'rb') as checkpoint:
            models_ema[cls_name].load_state_dict(
                torch.load(checkpoint, map_location='cpu')['model_ema']
            )

    return models_ema, parallel_models

def create_generator(cfg):
    return generator.Generator(
        latent_dim=512,
        scene_range=cfg.dataset_config.scene_range,
        attention_values=cfg.generator.attention_values,
        use_viewdir=cfg.use_viewdir,
        use_encoder=cfg.generator.use_encoder,
        disable_stylegan_noise=cfg.generator.disable_stylegan_noise,
        use_sdf=cfg.use_sdf,
        num_classes=None).to(DEVICE)

def run_classification(
    cfg, encoders, models_ema, generators,
    train_split, train_eval_split, test_split,
):
    feature_extractor = dino.ViTExtractor()
    if cfg.inv.save_images:
        assert isinstance(cfg.inv.every_n_steps, int)

    batch_size = 1
    test_bs = batch_size

    focal_guesses = pose_estimation.get_focal_guesses(train_eval_split.focal_length)

    random_generator = torch.Generator()
    random_generator.manual_seed(1234)
    n_images_fid = len(train_eval_split.images)
    remaining = n_images_fid
    train_eval_split.eval_indices = []
    while remaining > 0:
        nimg = len(train_eval_split.images)
        train_eval_split.eval_indices.append(
            torch.randperm(nimg, generator=random_generator)[:remaining])
        remaining -= len(train_eval_split.eval_indices[-1])
    train_eval_split.eval_indices = torch.cat(train_eval_split.eval_indices,
                                                dim=0).sort()[0]
    image_indices = train_eval_split.eval_indices

    checkpoint_steps = [0, cfg.inv.checkpoint_steps]

    idx = 0

    indexes = []
    paths = []
    target_classes = []
    predict_classes = []

    while idx < len(image_indices):
        t1 = time.time()

        if test_bs != 1 and image_indices[idx:idx + test_bs].shape[0] < test_bs:
            test_bs = 1

        target_img_idx = image_indices[idx:idx + test_bs]

        target_img = train_eval_split[target_img_idx].images
        target_cls = train_eval_split[target_img_idx].classes_id
        target_path = train_eval_split.get_path(target_img_idx)

        # Target for evaluation
        target_img_fid_ = train_eval_split[
            target_img_idx].images  # Cropped
        target_tform_cam2world = train_eval_split[target_img_idx].tform_cam2world
        target_focal = train_eval_split[target_img_idx].focal_length
        target_center = None
        target_bbox = None
        target_center_fid = train_eval_split[target_img_idx].center
        target_bbox_fid = train_eval_split[target_img_idx].bbox

        gt_cam2world_mat = target_tform_cam2world.clone()

        target_img_for_dino = target_img[:,:,:,:3].permute(0, 3, 1, 2) / 2 + 0.5
        target_img_for_dino = feature_extractor.normalize(target_img_for_dino)
        target_features = feature_extractor.extract_descriptors(target_img_for_dino)

        if cfg.inv.save_images and idx % cfg.inv.every_n_steps == 0:
            Path("resources/images/targets").mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(nrows=1, ncols=1)
            plt.imshow(target_img[0][:,:,:3].cpu() / 2 + 0.5)
            plt.axis("off")
            fig.savefig(f"resources/images/targets/target_{idx}.png")
            plt.close(fig)

        predicted_psnr_loss = []
        predicted_features = []
        predicted_class = []
        for cls_name in cfg.ImageNet:

            cls_encoder = encoders[cls_name]
            cls_generator = generators[cls_name]

            z_avg = models_ema[cls_name].mapping_network.get_average_w()
            z_ = z_avg.clone().expand(test_bs, -1, -1).contiguous()

            with torch.no_grad():

                coord_regressor_img = target_img[..., :3].permute(0, 3, 1, 2)
                if test_bs == 1:
                    target_coords, target_mask, target_w = cls_encoder.module(
                        coord_regressor_img)
                else:
                    target_coords, target_mask, target_w = cls_encoder(
                        coord_regressor_img)
                assert target_coords is not None

                estimated_cam2world_mat, estimated_focal, _ = estimate_poses_batch(
                    target_coords, target_mask, focal_guesses
                )
                target_tform_cam2world = estimated_cam2world_mat
                target_focal = estimated_focal 
                assert target_w is not None

                z_.data[:] = target_w
                z_ = z_.mean(dim=1, keepdim=True)

            z_ /= cfg.inv.lr_gain_z
            z_ = z_.requires_grad_()

            z0_, t2_, s_, R_ = pose_utils.matrix_to_pose(
                target_tform_cam2world,
                target_focal,
                camera_flipped=cfg.dataset_config.camera_flipped
            )
            # Optimize pose
            if not cfg.inv.no_optimize_pose:
                t2_.requires_grad_()
                s_.requires_grad_()
                R_.requires_grad_()
            if z0_ is not None:
                if not cfg.inv.no_optimize_pose:
                    z0_.requires_grad_()
                param_list = [z_, z0_, R_, s_, t2_]
            else:
                param_list = [z_, R_, s_, t2_]
            if cfg.inv.no_optimize_pose:
                param_list = param_list[:1]

            extra_model_inputs = {}
            optimizer = torch.optim.Adam(param_list, lr=2e-2, betas=(0.9, 0.95))
            grad_norms = []
            for _ in range(len(param_list)):
                grad_norms.append([])

            model_to_call = cls_generator if z_.shape[
                0] > 1 else cls_generator.module

            niter = max(checkpoint_steps)

            if 0 in checkpoint_steps:
                rgb_predicted = evaluate_inversion(
                    cfg,
                    model_to_call, cls_name,
                    z_ * cfg.inv.lr_gain_z, z0_, R_, s_, t2_,
                    target_center_fid, target_bbox_fid,
                )

            def optimize_iter(
                module, rgb_predicted, acc_predicted,
                semantics_predicted, extra_model_outputs, target_img,
                cam, focal
            ):

                target = target_img[..., :3]

                rgb_predicted_for_loss = rgb_predicted
                target_for_loss = target
                loss = 0.
                if cfg.inv.loss in ['vgg_nocrop', 'vgg', 'mixed']:
                    rgb_predicted_for_loss_aug = rgb_predicted_for_loss.permute(
                        0, 3, 1, 2)
                    target_for_loss_aug = target_for_loss.permute(0, 3, 1, 2)
                    loss = loss + module.lpips_net(
                        rgb_predicted_for_loss_aug, target_for_loss_aug
                    ).mean() * rgb_predicted.shape[
                        0]  # Disjoint samples, sum instead of average over batch
                if cfg.inv.loss in ['l1', 'mixed']:
                    loss = loss + F.l1_loss(rgb_predicted_for_loss, target_for_loss
                                        ) * rgb_predicted.shape[0]
                if cfg.inv.loss == 'mse':
                    loss = F.mse_loss(rgb_predicted_for_loss,
                                    target_for_loss) * rgb_predicted.shape[0]

                if cfg.inv.loss == 'mixed':
                    loss = loss / 2  # Average L1 and VGG

                with torch.no_grad():
                    psnr_monitor = metrics.psnr(rgb_predicted[..., :3] / 2 + 0.5,
                                                target[..., :3] / 2 + 0.5)
                    lpips_monitor = module.lpips_net(
                        rgb_predicted[..., :3].permute(0, 3, 1, 2),
                        target[..., :3].permute(0, 3, 1, 2),
                        normalize=False)

                return loss, psnr_monitor, lpips_monitor, rgb_predicted

            for it in range(niter):
                cam, focal = pose_utils.pose_to_matrix(
                    z0_,
                    t2_,
                    s_,
                    F.normalize(R_, dim=-1),
                    camera_flipped=cfg.dataset_config['camera_flipped'])

                loss, psnr_monitor, lpips_monitor, rgb_predicted = model_to_call(
                    cam,
                    focal,
                    target_center,
                    target_bbox,
                    z_ * cfg.inv.lr_gain_z,
                    closure=optimize_iter,
                    closure_params={
                        'target_img': target_img,
                        'cam': cam,
                        'focal': focal
                    }
                )
                loss = loss.sum()
                psnr_monitor = psnr_monitor.mean()
                lpips_monitor = lpips_monitor.mean()

                loss.backward()
                for i, param in enumerate(param_list):
                    if param.grad is not None:
                        grad_norms[i].append(param.grad.norm().item())
                    else:
                        grad_norms[i].append(0.)
                optimizer.step()
                optimizer.zero_grad()
                R_.data[:] = F.normalize(R_.data, dim=-1)
                if z0_ is not None:
                    z0_.data.clamp_(-4, 4)
                s_.data.abs_()

                rgb_predicted = evaluate_inversion(
                    cfg,
                    model_to_call, cls_name,
                    z_ * cfg.inv.lr_gain_z, z0_, R_, s_, t2_,
                    target_center_fid, target_bbox_fid,
                )

            target_img = target_img[..., :3]

            # Calculate psnr loss
            predicted_psnr_loss.append(
                metrics.psnr(target_img / 2 + 0.5, rgb_predicted / 2 + 0.5)
            )
            
            # Calculate features using DINO
            rgb_predicted_for_dino = rgb_predicted[:,:,:,:3].permute(0, 3, 1, 2) / 2 + 0.5
            rgb_predicted_for_dino = feature_extractor.normalize(rgb_predicted_for_dino)
            rgb_predicted_features = feature_extractor.extract_descriptors(rgb_predicted_for_dino)
            predicted_features.append(rgb_predicted_features)

            # Add class
            predicted_class.append(cls_name)

            # Save image
            if cfg.inv.save_images and idx % cfg.inv.every_n_steps == 0:
                Path(f"resources/images/{cls_name}").mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(nrows=1, ncols=1)
                plt.imshow(rgb_predicted[0].cpu().clamp(-1, 1) / 2 + 0.5)
                plt.axis("off")
                plt.savefig(f"resources/images/{cls_name}/{cls_name}_{idx}.png")
                plt.close(fig)
        
        predicted_features = torch.cat(predicted_features)
        cos_sim = chunk_cosine_sim(target_features, predicted_features)[0]
        predict_cls = train_eval_split.class_to_id[
            predicted_class[torch.argmax(cos_sim).item()]
        ]
        #predict_cls = max(zip(predicted_psnr_loss, predicted_class))[1]
        
        indexes.append(idx)
        paths.append(target_path)
        target_classes.append(target_cls.cpu().item())
        predict_classes.append(predict_cls.cpu().item())

        t2 = time.time()
        idx += test_bs
        print(
            f'[{idx}/{len(image_indices)}] Finished batch in {t2-t1} s ({(t2-t1)/test_bs} s/img)'
        )

    return target_classes, predict_classes, indexes, paths

def estimate_poses_batch(target_coords, target_mask, focal_guesses):
    target_mask = target_mask > 0.9
    if focal_guesses is None:
        # Use a large focal length to approximate ortho projection
        is_ortho = True
        focal_guesses = [100.]
    else:
        is_ortho = False

    world2cam_mat, estimated_focal, errors = pose_estimation.compute_pose_pnp(
        target_coords.cpu().numpy(),
        target_mask.cpu().numpy(), focal_guesses)

    if is_ortho:
        # Convert back to ortho
        s = 2 * focal_guesses[0] / -world2cam_mat[:, 2, 3]
        t2 = world2cam_mat[:, :2, 3] * s[..., None]
        world2cam_mat_ortho = world2cam_mat.copy()
        world2cam_mat_ortho[:, :2, 3] = t2
        world2cam_mat_ortho[:, 2, 3] = -10.
        world2cam_mat = world2cam_mat_ortho

    estimated_cam2world_mat = pose_utils.invert_space(
        torch.from_numpy(world2cam_mat).float()).to(target_coords.device)
    estimated_focal = torch.from_numpy(estimated_focal).float().to(
        target_coords.device)
    if is_ortho:
        estimated_cam2world_mat /= torch.from_numpy(
            s[:, None, None]).float().to(estimated_cam2world_mat.device)
        estimated_focal = None

    return estimated_cam2world_mat, estimated_focal, errors

def evaluate_inversion(
    cfg,
    model_to_call, cls_name,
    z_, z0_, R_, s_, t2_,
    target_center_fid, target_bbox_fid,
):

    # Compute metrics for report
    cam, focal = pose_utils.pose_to_matrix(
        z0_.detach() if z0_ is not None else None,
        t2_.detach(),
        s_.detach(),
        F.normalize(R_.detach(), dim=-1),
        camera_flipped=cfg.dataset_config.camera_flipped
    )
    rgb_predicted, _, __, ___, ____, _____ = model_to_call(
        cam,
        focal,
        target_center_fid,
        target_bbox_fid,
        z_.detach()
    )
    return rgb_predicted


def main():
    cfg = OmegaConf.load("config.yaml")
    encoders = load_encoders(cfg)
    models_ema, generators = load_generators(cfg)
    _, train_split, train_eval_split, test_split = imagenet_loader.load_dataset(cfg, DEVICE)
    targets, predicts, idxs, paths = run_classification(
        cfg, encoders, models_ema, generators,
        train_split, train_eval_split, test_split,
    )
    Path("./resources/results").mkdir(parents=True, exist_ok=True)
    with open("./resources/results/targets.npy", 'wb') as f:
        np.save(f, np.array(targets))
    with open("./resources/results/predicts.npy", 'wb') as f:
        np.save(f, np.array(predicts))
    with open("./resources/results/indexes.npy", 'wb') as f:
        np.save(f, np.array(idxs))
    with open("./resources/results/paths.npy", 'wb') as f:
        np.save(f, np.array(paths))

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape (B_x)x1xtxd' where d' is the dimensionality of the descriptors and t
    is the number of tokens.
    :param y: a tensor of descriptors of shape (B_y)x1xtxd' where d' is the dimensionality of the descriptors and t
    is the number of tokens.
    :return: cosine similarity between all vectors in x and all vectors in y. Has shape of (B_x)x(B_y) """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm = x / x.norm(dim=1)[:, None]
    y_norm = y / y.norm(dim=1)[:, None]
    res = torch.mm(x_norm, y_norm.transpose(0,1))
    return res

if __name__ == "__main__":
    main()
