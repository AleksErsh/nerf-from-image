# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import arguments
from data import loaders
from lib import pose_utils
from lib import nerf_utils
from lib import utils
from lib import fid
from lib import ops
from lib import metrics
from lib import pose_estimation
from models import generator
from models import discriminator
from models import encoder

args = arguments.parse_args()
gpu_ids = list(range(args.gpus))

if args.gpus > 0 and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

print(args)

experiment_name = arguments.suggest_experiment_name(args)
resume_from = None
log_dir = 'gan_logs'
report_dir = 'reports'
file_dir = 'gan_checkpoints'

checkpoint_dir = os.path.join(args.root_path, file_dir, experiment_name)
print('Saving checkpoints to', checkpoint_dir)

tensorboard_dir = os.path.join(args.root_path, log_dir, experiment_name)
report_dir = os.path.join(args.root_path, report_dir)
print('Saving tensorboard logs to', tensorboard_dir)
print('Saving inversion reports to', report_dir)
utils.mkdir(report_dir)

if args.resume_from:
    print('Attempting to load latest checkpoint...')
    last_checkpoint_dir = os.path.join(args.root_path, 'gan_checkpoints',
                                        args.resume_from,
                                        'checkpoint_latest.pth')

    if utils.file_exists(last_checkpoint_dir):
        print('Resuming from manual checkpoint', last_checkpoint_dir)
        with utils.open_file(last_checkpoint_dir, 'rb') as f:
            resume_from = torch.load(f, map_location='cpu')
    else:
        raise ValueError(
            f'Specified checkpoint {args.resume_from} does not exist!')

if args.attention_values > 0:
    color_palette = utils.get_color_palette(args.attention_values).to(device)
else:
    color_palette = None

dataset_config, train_split, train_eval_split, test_split = loaders.load_dataset(
    args, device)


def render(target_model,
           height,
           width,
           tform_cam2world,
           focal_length,
           center,
           bbox,
           model_input,
           depth_samples_per_ray,
           randomize=True,
           compute_normals=False,
           compute_semantics=False,
           compute_coords=False,
           extra_model_outputs=[],
           extra_model_inputs={},
           force_no_cam_grad=False):

    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(),
            dataset_config['scene_range'])

    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins,
        ray_directions,
        near_thresh,
        far_thresh,
        depth_samples_per_ray,
        randomize=randomize,
    )

    if force_no_cam_grad:
        query_points = query_points.detach()
        depth_values = depth_values.detach()
        ray_directions = ray_directions.detach()

    if args.use_viewdir:
        viewdirs = ray_directions.unsqueeze(-2)
    else:
        viewdirs = None

    model_outputs = target_model(viewdirs, model_input,
                                 ['sampler'] + extra_model_outputs,
                                 extra_model_inputs)
    radiance_field_sampler = model_outputs['sampler']
    del model_outputs['sampler']

    request_sampler_outputs = ['sigma', 'rgb']
    if compute_normals:
        assert args.use_sdf
        request_sampler_outputs.append('normals')
    if compute_semantics:
        assert args.attention_values > 0
        request_sampler_outputs.append('semantics')
    if compute_coords:
        request_sampler_outputs.append('coords')
    sampler_outputs_coarse = radiance_field_sampler(query_points,
                                                    request_sampler_outputs)
    sigma = sampler_outputs_coarse['sigma'].view(*query_points.shape[:-1], -1)
    rgb = sampler_outputs_coarse['rgb'].view(*query_points.shape[:-1], -1)

    if compute_normals:
        normals = sampler_outputs_coarse['normals'].view(
            *query_points.shape[:-1], -1)
    else:
        normals = None

    if compute_semantics:
        semantics = sampler_outputs_coarse['semantics'].view(
            *query_points.shape[:-1], -1)
    else:
        semantics = None

    if compute_coords:
        coords = sampler_outputs_coarse['coords'].view(*query_points.shape[:-1],
                                                       -1)
    else:
        coords = None

    if args.fine_sampling:
        z_vals = depth_values
        with torch.no_grad():
            weights = nerf_utils.render_volume_density_weights_only(
                sigma.squeeze(-1), ray_origins, ray_directions,
                depth_values).flatten(0, 2)

            # Smooth weights as in EG3D
            weights = F.max_pool1d(weights.unsqueeze(1).float(),
                                   2,
                                   1,
                                   padding=1)
            weights = F.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = nerf_utils.sample_pdf(
                z_vals_mid.flatten(0, 2),
                weights[..., 1:-1],
                depth_samples_per_ray,
                deterministic=not randomize,
            )
            z_samples = z_samples.view(*z_vals.shape[:3], z_samples.shape[-1])

        z_values_sorted, z_indices_sorted = torch.sort(torch.cat(
            (z_vals, z_samples), dim=-1),
                                                       dim=-1)
        query_points_fine = ray_origins[
            ...,
            None, :] + ray_directions[..., None, :] * z_samples[..., :, None]

        sampler_outputs_fine = radiance_field_sampler(query_points_fine,
                                                      request_sampler_outputs)
        sigma_fine = sampler_outputs_fine['sigma'].view(
            *query_points_fine.shape[:-1], -1)
        rgb_fine = sampler_outputs_fine['rgb'].view(
            *query_points_fine.shape[:-1], -1)
        if compute_normals:
            normals_fine = sampler_outputs_fine['normals'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            normals_fine = None
        if compute_semantics:
            semantics_fine = sampler_outputs_fine['semantics'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            semantics_fine = None
        if compute_coords:
            coords_fine = sampler_outputs_fine['coords'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            coords_fine = None

        sigma = torch.cat((sigma, sigma_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  sigma.shape[-1]))
        rgb = torch.cat((rgb, rgb_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  rgb.shape[-1]))
        if normals_fine is not None:
            normals = torch.cat((normals, normals_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      normals.shape[-1]))
        if semantics_fine is not None:
            semantics = torch.cat((semantics, semantics_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      semantics.shape[-1]))
        if coords_fine is not None:
            coords = torch.cat((coords, coords_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      coords.shape[-1]))
        depth_values = z_values_sorted

    if coords is not None:
        semantics = coords

    rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted = nerf_utils.render_volume_density(
        sigma.squeeze(-1),
        rgb,
        ray_origins,
        ray_directions,
        depth_values,
        normals,
        semantics,
        white_background=dataset_config['white_background'])

    return rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted, model_outputs


class GANLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, target_positive: bool):
        if target_positive:
            return F.softplus(-x).mean()
        else:
            return F.softplus(x).mean()


evaluation_res = args.resolution
inception_net = fid.init_inception('tensorflow').to(device).eval()

# Compute stats
print('Computing FID stats...')


def compute_real_fid_stats(images_fid_actual):
    n_images_fid = len(images_fid_actual)
    all_activations = []
    with torch.no_grad():
        test_bs = args.batch_size
        for idx in tqdm(range(0, n_images_fid, test_bs)):
            im = images_fid_actual[idx:idx + test_bs].to(device) / 2 + 0.5
            im = im.permute(0, 3, 1, 2)[:, :3]
            if evaluation_res == 64:
                im = F.avg_pool2d(im, 2)
            assert im.shape[-1] == evaluation_res
            all_activations.append(
                fid.forward_inception_batch(inception_net, im))
    all_activations = np.concatenate(all_activations, axis=0)
    return fid.calculate_stats(all_activations)


train_eval_split.fid_stats = compute_real_fid_stats(train_eval_split.images)
n_images_fid = len(train_eval_split.images)

if ((args.run_inversion and args.inv_use_testset) or
        args.use_encoder) and dataset_config['views_per_object_test']:
    print('Computing FID stats for test set...')
    test_split.fid_stats = compute_real_fid_stats(test_split.images)

random_seed = 1234
n_images_fid_max = 8000  # Matches Pix2NeRF evaluation protocol

random_generator = torch.Generator()
random_generator.manual_seed(random_seed)
if n_images_fid > n_images_fid_max:
    # Select random indices without replacement
    train_eval_split.eval_indices = torch.randperm(
        n_images_fid, generator=random_generator)[:n_images_fid_max].sort()[0]
else:
    if args.dataset.startswith('imagenet_'):
        # Select n_images_fid random poses (potentially repeated)
        remaining = n_images_fid
        train_eval_split.eval_indices = []
        while remaining > 0:
            nimg = len(train_eval_split.images)
            train_eval_split.eval_indices.append(
                torch.randperm(nimg, generator=random_generator)[:remaining])
            remaining -= len(train_eval_split.eval_indices[-1])
        train_eval_split.eval_indices = torch.cat(train_eval_split.eval_indices,
                                                  dim=0).sort()[0]
    else:
        assert len(train_split.images) == n_images_fid or len(
            train_split.images) == 2 * n_images_fid
        train_eval_split.eval_indices = torch.arange(n_images_fid)

print(f'Evaluating training FID on {len(train_eval_split.eval_indices)} images')

if args.use_encoder or args.run_inversion:

    def compute_view_perm(target_img_indices, views_per_object):
        target_img_perm = torch.randperm(len(target_img_indices),
                                             generator=random_generator)
        target_img_perm = torch.LongTensor(target_img_perm)
        return target_img_perm

    train_eval_split.eval_indices_perm = compute_view_perm(
        train_eval_split.eval_indices, dataset_config['views_per_object'])

random_generator.manual_seed(random_seed)  # Re-seed
z_fid_untrunc = torch.randn(
    (len(train_eval_split.eval_indices), args.latent_dim),
    generator=random_generator)
z_fid_untrunc = z_fid_untrunc.to(device)

torch.cuda.empty_cache()

batch_size = args.batch_size
use_viewdir = args.use_viewdir
supervise_alpha = args.supervise_alpha
use_encoder = args.use_encoder

# Number of depth samples along each ray
depth_samples_per_ray = 64
if not args.fine_sampling:
    depth_samples_per_ray *= 2  # More fair comparison

num_iters = 0 if args.run_inversion else args.iterations

def create_model():
    return generator.Generator(
        args.latent_dim,
        dataset_config['scene_range'],
        attention_values=args.attention_values,
        use_viewdir=use_viewdir,
        use_encoder=args.use_encoder,
        disable_stylegan_noise=args.disable_stylegan_noise,
        use_sdf=args.use_sdf,
        num_classes=train_split.num_classes
        if args.use_class else None).to(device)


if args.dual_discriminator_l1 or args.dual_discriminator_mse:
    discriminator = None
else:
    discriminator = discriminator.Discriminator(
        args.resolution,
        nc=4 if supervise_alpha else 3,
        dataset_config=dataset_config,
        conditional_pose=args.conditional_pose,
        use_encoder=args.use_encoder,
        num_classes=train_split.num_classes if args.use_class else None,
    ).to(device)
discriminator_list = [discriminator]
if args.dual_discriminator:
    if args.use_encoder:
        # Instantiate another discriminator
        discriminator2 = discriminator.Discriminator(
            args.resolution,
            nc=4 if supervise_alpha else 3,
            dataset_config=dataset_config,
            conditional_pose=args.conditional_pose,
            num_classes=train_split.num_classes if args.use_class else None,
            use_encoder=False).to(device)
    else:
        discriminator2 = discriminator
    discriminator_list.append(discriminator2)


class ParallelModel(nn.Module):

    def __init__(self, resolution, model=None, model_ema=None, lpips_net=None):
        super().__init__()
        self.resolution = resolution
        self.model = model
        self.model_ema = model_ema
        self.lpips_net = lpips_net

    def forward(self,
                tform_cam2world,
                focal,
                center,
                bbox,
                c,
                use_ema=False,
                ray_multiplier=1,
                res_multiplier=1,
                pretrain_sdf=False,
                compute_normals=False,
                compute_semantics=False,
                compute_coords=False,
                encoder_output=False,
                closure=None,
                closure_params=None,
                extra_model_outputs=[],
                extra_model_inputs={},
                force_no_cam_grad=False):
        model_to_use = self.model_ema if use_ema else self.model
        if pretrain_sdf:
            return model_to_use(
                None,
                c,
                request_model_outputs=['sdf_distance_loss', 'sdf_eikonal_loss'])
        if encoder_output:
            return model_to_use.emb(c)

        output = render(model_to_use,
                        int(self.resolution * res_multiplier),
                        int(self.resolution * res_multiplier),
                        tform_cam2world,
                        focal,
                        center,
                        bbox,
                        c,
                        depth_samples_per_ray * ray_multiplier,
                        compute_normals=compute_normals,
                        compute_semantics=compute_semantics,
                        compute_coords=compute_coords,
                        extra_model_outputs=extra_model_outputs,
                        extra_model_inputs=extra_model_inputs,
                        force_no_cam_grad=force_no_cam_grad)
        if closure is not None:
            return closure(
                self, output[0], output[2], output[4], output[-1],
                **closure_params)  # RGB, alpha, semantics, extra_outptus
        else:
            return output


if args.use_encoder or args.run_inversion:
    loss_fn_lpips = metrics.LPIPSLoss().to(device)
else:
    loss_fn_lpips = None

if args.run_inversion:
    model = None
else:
    model = create_model()

model_ema = create_model()
model_ema.eval()
model_ema.requires_grad_(False)
if model is not None:
    model_ema.load_state_dict(model.state_dict())

parallel_model = nn.DataParallel(
    ParallelModel(args.resolution,
                  model=model,
                  model_ema=model_ema,
                  lpips_net=loss_fn_lpips), gpu_ids).to(device)
parallel_discriminator_list = [
    nn.DataParallel(d, gpu_ids) if d is not None else None
    for d in discriminator_list
]

total_params = 0
for param in model_ema.parameters():
    total_params += param.numel()
print('Params G:', total_params / 1000000, 'M')

for d_idx, d in enumerate(discriminator_list):
    if d is None:
        print(f'Params D_{d_idx}: none')
    else:
        total_params = 0
        for param in d.parameters():
            total_params += param.numel()
        print(f'Params D_{d_idx}:', total_params / 1000000, 'M')

criterion = GANLoss()

# These are set to True dynamically during runtime to optimize memory usage
[d.requires_grad_(False) for d in discriminator_list if d is not None]
if model is not None:
    model.requires_grad_(False)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
rng = np.random.RandomState(random_seed)
train_sampler = utils.EndlessSampler(len(train_split.images), rng)

# Seed CUDA RNGs separately
seed_generator = np.random.RandomState(random_seed)
for device_id in gpu_ids:
    with torch.cuda.device(device_id):
        gpu_seed = int.from_bytes(np.random.bytes(4), 'little', signed=False)
        torch.cuda.manual_seed(gpu_seed)

i = 0
best_fid = 1000
augment_p_effective = 0.
ppl_running_avg = None


if resume_from is not None:
    print('Loading specified checkpoint...')
    if model is not None and 'model' in resume_from:
        model.load_state_dict(resume_from['model'])
    model_ema.load_state_dict(resume_from['model_ema'])
    if 'iteration' in resume_from:
        i = resume_from['iteration']
        print('Resuming from iteration', i)
    else:
        i = args.iterations

    if 'random_state' in resume_from:
        print('Restoring RNG state...')
        utils.restore_random_state(resume_from['random_state'], train_sampler,
                                   rng, gpu_ids)

    if 'lr_g' in resume_from:
        lr_g = resume_from['lr_g']
    if 'lr_d' in resume_from:
        lr_d = resume_from['lr_d']
    if 'best_fid' in resume_from:
        best_fid = resume_from['best_fid']
    if 'augment_p_effective' in resume_from:
        augment_p_effective = resume_from['augment_p']
    if args.path_length_regularization and 'ppl_running_avg' in resume_from:
        ppl_running_avg = resume_from['ppl_running_avg']

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


if args.run_inversion:
    # Global config
    use_testset = args.inv_use_testset
    use_pose_regressor = True
    use_latent_regressor = True
    loss_to_use = args.inv_loss
    lr_gain_z = args.inv_gain_z
    inv_no_split = args.inv_no_split
    no_optimize_pose = args.inv_no_optimize_pose

    batch_size = args.batch_size // 4 * len(gpu_ids)

    if args.dataset == 'p3d_car' and use_testset:
        split_str = 'imagenettest' if args.inv_use_imagenet_testset else 'test'
    else:
        split_str = 'test' if use_testset else 'train'
    if args.inv_use_separate:
        mode_str = '_separate'
    else:
        mode_str = '_joint'
    if no_optimize_pose:
        mode_str += '_nooptpose'
    else:
        mode_str += '_optpose'
    w_split_str = 'nosplit' if inv_no_split else 'split'
    cfg_xid = f'_{args.xid}' if len(args.xid) > 0 else ''
    cfg_string = f'i{cfg_xid}_{split_str}{mode_str}_{loss_to_use}_gain{lr_gain_z}_{w_split_str}'
    cfg_string += f'_it{resume_from["iteration"]}'

    print('Config string:', cfg_string)

    report_dir_effective = os.path.join(report_dir, args.resume_from,
                                        cfg_string)
    print('Saving report in', report_dir_effective)
    utils.mkdir(report_dir_effective)
    writer = tensorboard.SummaryWriter(report_dir_effective)

    print('Resuming from pose regressor', args.coord_resume_from)
    coord_regressor = encoder.BootstrapEncoder(
        args.latent_dim,
        pose_regressor=use_pose_regressor,
        latent_regressor=use_latent_regressor,
        separate_backbones=args.inv_use_separate,
        pretrained=False).to(device)

    coord_regressor = nn.DataParallel(coord_regressor, gpu_ids)
    checkpoint_path = os.path.join(args.root_path, 'coords_checkpoints',
                                    args.resume_from,
                                    f'{args.coord_resume_from}.pth')
    with utils.open_file(checkpoint_path, 'rb') as f:
        coord_regressor.load_state_dict(
            torch.load(f, map_location='cpu')['model_coord'])
    coord_regressor.requires_grad_(False)
    coord_regressor.eval()

    focal_guesses = pose_estimation.get_focal_guesses(
        train_split.focal_length)

    image_indices = test_split.eval_indices if use_testset else train_eval_split.eval_indices
    image_indices_perm = test_split.eval_indices_perm if use_testset else train_eval_split.eval_indices_perm

    if args.inv_encoder_only:
        checkpoint_steps = [0]
    elif lr_gain_z >= 10:
        checkpoint_steps = [0, 10]
    else:
        checkpoint_steps = [0, 30]

    report = {
        step: {
            'ws': [],
            'z0': [],
            'R': [],
            's': [],
            't2': [],
            'psnr': [],
            'psnr_random': [],
            'lpips': [],
            'lpips_random': [],
            'ssim': [],
            'ssim_random': [],
            'iou': [],
            'rot_error': [],
            'inception_activations_front': [],  # Front view
            'inception_activations_random': [],  # Random view
        } for step in checkpoint_steps
    }

    with torch.no_grad():
        z_avg = model_ema.mapping_network.get_average_w()

    idx = 0
    test_bs = batch_size

    report_checkpoint_path = os.path.join(report_dir_effective,
                                          'report_checkpoint.pth')
    if utils.file_exists(report_checkpoint_path):
        print('Found inversion report checkpoint in', report_checkpoint_path)
        with utils.open_file(report_checkpoint_path, 'rb') as f:
            report_checkpoint = torch.load(f)
            report = report_checkpoint['report']
            idx = report_checkpoint['idx']
            test_bs = report_checkpoint['test_bs']
    else:
        print('Inversion report checkpoint not found, starting from scratch...')

    while idx < len(image_indices):
        t1 = time.time()
        frames = []

        if test_bs != 1 and image_indices[idx:idx + test_bs].shape[0] < test_bs:
            test_bs = 1

        target_img_idx = image_indices[idx:idx + test_bs]
        target_img_idx_perm = image_indices_perm[idx:idx + test_bs]
        if use_testset:
            target_img = test_split[
                target_img_idx].images  # Target for optimization (always cropped)
            target_img_fid_ = target_img  # Target for evaluation (front view -- always cropped)
            target_tform_cam2world = test_split[target_img_idx].tform_cam2world
            target_focal = test_split[target_img_idx].focal_length
            target_center = None
            target_bbox = None
            views_per_object = dataset_config['views_per_object_test']
            if views_per_object > 1:
                target_img_fid_random_ = test_split[target_img_idx_perm].images

            if use_pose_regressor:
                target_center_fid = None
                target_bbox_fid = None
            else:
                target_center_fid = test_split[target_img_idx].center
                target_bbox_fid = test_split[target_img_idx].bbox

            target_tform_cam2world_perm = test_split[
                target_img_idx_perm].tform_cam2world
            target_focal_perm = test_split[target_img_idx_perm].focal_length
            target_center_perm = test_split[target_img_idx_perm].center
            target_bbox_perm = test_split[target_img_idx_perm].bbox
        else:
            target_img = train_split[target_img_idx].images
            views_per_object = dataset_config['views_per_object']

            # Target for evaluation
            target_img_fid_ = train_eval_split[
                target_img_idx].images  # Cropped
            target_tform_cam2world = train_split[target_img_idx].tform_cam2world
            target_focal = train_split[target_img_idx].focal_length
            target_center = None
            target_bbox = None
            target_center_fid = train_eval_split[target_img_idx].center
            target_bbox_fid = train_eval_split[target_img_idx].bbox

            target_tform_cam2world_perm = train_eval_split[
                target_img_idx_perm].tform_cam2world
            target_focal_perm = train_eval_split[
                target_img_idx_perm].focal_length
            target_center_perm = train_eval_split[target_img_idx_perm].center
            target_bbox_perm = train_eval_split[target_img_idx_perm].bbox

        gt_cam2world_mat = target_tform_cam2world.clone()
        z_ = z_avg.clone().expand(test_bs, -1, -1).contiguous()

        with torch.no_grad():
            coord_regressor_img = target_img[..., :3].permute(0, 3, 1, 2)
            if test_bs == 1:
                target_coords, target_mask, target_w = coord_regressor.module(
                    coord_regressor_img)
            else:
                target_coords, target_mask, target_w = coord_regressor(
                    coord_regressor_img)
        if use_pose_regressor:
            assert target_coords is not None
            estimated_cam2world_mat, estimated_focal, _ = estimate_poses_batch(
                target_coords, target_mask, focal_guesses)
            target_tform_cam2world = estimated_cam2world_mat
            target_focal = estimated_focal

        assert target_w is not None
        z_.data[:] = target_w

        if inv_no_split:
            z_ = z_.mean(dim=1, keepdim=True)

        z_ /= lr_gain_z
        z_ = z_.requires_grad_()

        z0_, t2_, s_, R_ = pose_utils.matrix_to_pose(
            target_tform_cam2world,
            target_focal,
            camera_flipped=dataset_config['camera_flipped'])
        if not no_optimize_pose:
            t2_.requires_grad_()
            s_.requires_grad_()
            R_.requires_grad_()
        if z0_ is not None:
            if not no_optimize_pose:
                z0_.requires_grad_()
            param_list = [z_, z0_, R_, s_, t2_]
            param_names = ['z', 'f', 'R', 's', 't']
        else:
            param_list = [z_, R_, s_, t2_]
            param_names = ['z', 'R', 's', 't']
        if no_optimize_pose:
            param_list = param_list[:1]
            param_names = param_names[:1]

        extra_model_inputs = {}
        optimizer = torch.optim.Adam(param_list, lr=2e-3, betas=(0.9, 0.95))
        grad_norms = []
        for _ in range(len(param_list)):
            grad_norms.append([])

        model_to_call = parallel_model if z_.shape[
            0] > 1 else parallel_model.module

        psnrs = []
        lpipss = []
        rot_errors = []
        niter = max(checkpoint_steps)

        def evaluate_inversion(it):
            item = report[it]
            item['ws'].append(z_.detach().cpu() * lr_gain_z)
            if z0_ is not None:
                item['z0'].append(z0_.detach().cpu())
            item['R'].append(R_.detach().cpu())
            item['s'].append(s_.detach().cpu())
            item['t2'].append(t2_.detach().cpu())

            # Compute metrics for report
            cam, focal = pose_utils.pose_to_matrix(
                z0_.detach() if z0_ is not None else None,
                t2_.detach(),
                s_.detach(),
                F.normalize(R_.detach(), dim=-1),
                camera_flipped=dataset_config['camera_flipped'])
            rgb_predicted, _, acc_predicted, normals_predicted, semantics_predicted, extra_model_outputs = model_to_call(
                cam,
                focal,
                target_center_fid,
                target_bbox_fid,
                z_.detach() * lr_gain_z,
                use_ema=True,
                compute_normals=args.use_sdf and idx == 0,
                compute_semantics=args.attention_values > 0,
                force_no_cam_grad=True,
                extra_model_outputs=['attention_values']
                if args.attention_values > 0 else [],
                extra_model_inputs={
                    k: v.detach() for k, v in extra_model_inputs.items()
                },
            )

            rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                                2).clamp_(
                                                                    -1, 1)
            target_perm = target_img_fid_.permute(0, 3, 1, 2)
            item['psnr'].append(
                metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
                             target_perm[:, :3] / 2 + 0.5,
                             reduction='none').cpu())
            item['ssim'].append(
                metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
                             target_perm[:, :3] / 2 + 0.5,
                             reduction='none').cpu())
            if dataset_config['has_mask']:
                item['iou'].append(
                    metrics.iou(acc_predicted,
                                target_perm[:, 3],
                                reduction='none').cpu())
            item['lpips'].append(
                loss_fn_lpips(rgb_predicted_perm[:, :3],
                              target_perm[:, :3],
                              normalize=False).flatten().cpu())
            item['inception_activations_front'].append(
                torch.FloatTensor(
                    fid.forward_inception_batch(
                        inception_net, rgb_predicted_perm[:, :3] / 2 + 0.5)))
            if not (args.dataset == 'p3d_car' and use_testset):
                # Ground-truth poses are not available on P3D Car (test set)
                item['rot_error'].append(
                    pose_utils.rotation_matrix_distance(cam, gt_cam2world_mat))

            if writer is not None and idx == 0:
                if it == checkpoint_steps[0]:
                    writer.add_images(f'img/ref',
                                      target_perm[:, :3].cpu() / 2 + 0.5, i)
                writer.add_images('img/recon_front',
                                  rgb_predicted_perm.cpu() / 2 + 0.5, it)
                writer.add_images('img/mask_front',
                                  acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
                                  it)
                if normals_predicted is not None:
                    writer.add_images(
                        'img/normals_front',
                        normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
                        it)
                if semantics_predicted is not None:
                    writer.add_images(
                        'img/semantics_front',
                        (semantics_predicted @ color_palette).cpu().permute(
                            0, 3, 1, 2) / 2 + 0.5, it)

            # Test with random poses
            rgb_predicted, _, _, normals_predicted, semantics_predicted, _ = model_to_call(
                target_tform_cam2world_perm,
                target_focal_perm,
                target_center_perm,
                target_bbox_perm,
                z_.detach() * lr_gain_z,
                use_ema=True,
                compute_normals=args.use_sdf and idx == 0,
                compute_semantics=args.attention_values > 0 and idx == 0,
                force_no_cam_grad=True,
                extra_model_inputs={
                    k: v.detach() for k, v in extra_model_inputs.items()
                },
            )
            rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                                2).clamp(-1, 1)
            if views_per_object > 1:
                target_perm_random = target_img_fid_random_.permute(0, 3, 1, 2)
                item['psnr_random'].append(
                    metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
                                 target_perm_random[:, :3] / 2 + 0.5,
                                 reduction='none').cpu())
                item['ssim_random'].append(
                    metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
                                 target_perm_random[:, :3] / 2 + 0.5,
                                 reduction='none').cpu())
                item['lpips_random'].append(
                    loss_fn_lpips(rgb_predicted_perm[:, :3],
                                  target_perm_random[:, :3],
                                  normalize=False).flatten().cpu())
            item['inception_activations_random'].append(
                torch.FloatTensor(
                    fid.forward_inception_batch(
                        inception_net, rgb_predicted_perm[:, :3] / 2 + 0.5)))
            if writer is not None and idx == 0:
                writer.add_images('img/recon_random',
                                  rgb_predicted_perm.cpu() / 2 + 0.5, it)
                writer.add_images('img/mask_random',
                                  acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
                                  it)
                if normals_predicted is not None:
                    writer.add_images(
                        'img/normals_random',
                        normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
                        it)
                if semantics_predicted is not None:
                    writer.add_images(
                        'img/semantics_random',
                        (semantics_predicted @ color_palette).cpu().permute(
                            0, 3, 1, 2) / 2 + 0.5, it)

        if 0 in checkpoint_steps:
            evaluate_inversion(0)

        def optimize_iter(module, rgb_predicted, acc_predicted,
                          semantics_predicted, extra_model_outputs, target_img,
                          cam, focal):

            target = target_img[..., :3]

            rgb_predicted_for_loss = rgb_predicted
            target_for_loss = target
            loss = 0.
            if loss_to_use in ['vgg_nocrop', 'vgg', 'mixed']:
                rgb_predicted_for_loss_aug = rgb_predicted_for_loss.permute(
                    0, 3, 1, 2)
                target_for_loss_aug = target_for_loss.permute(0, 3, 1, 2)
                loss = loss + module.lpips_net(
                    rgb_predicted_for_loss_aug, target_for_loss_aug
                ).mean() * rgb_predicted.shape[
                    0]  # Disjoint samples, sum instead of average over batch
            if loss_to_use in ['l1', 'mixed']:
                loss = loss + F.l1_loss(rgb_predicted_for_loss, target_for_loss
                                       ) * rgb_predicted.shape[0]
            if loss_to_use == 'mse':
                loss = F.mse_loss(rgb_predicted_for_loss,
                                  target_for_loss) * rgb_predicted.shape[0]

            if loss_to_use == 'mixed':
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
                camera_flipped=dataset_config['camera_flipped'])

            loss, psnr_monitor, lpips_monitor, rgb_predicted = model_to_call(
                cam,
                focal,
                target_center,
                target_bbox,
                z_ * lr_gain_z,
                use_ema=True,
                ray_multiplier=1 if args.fine_sampling else 4,
                res_multiplier=1,
                compute_normals=False and args.use_sdf,
                force_no_cam_grad=no_optimize_pose,
                closure=optimize_iter,
                closure_params={
                    'target_img': target_img,
                    'cam': cam,
                    'focal': focal
                },
                extra_model_inputs=extra_model_inputs,
            )
            normal_map = None
            loss = loss.sum()
            psnr_monitor = psnr_monitor.mean()
            lpips_monitor = lpips_monitor.mean()
            if writer is not None and idx == 0:
                writer.add_scalar('monitor_b0/psnr', psnr_monitor.item(), it)
                writer.add_scalar('monitor_b0/lpips', lpips_monitor.item(), it)
                rot_error = pose_utils.rotation_matrix_distance(
                    cam, gt_cam2world_mat).mean().item()
                rot_errors.append(rot_error)
                writer.add_scalar('monitor_b0/rot_error', rot_error, it)

            if args.use_sdf and normal_map is not None:
                rgb_predicted = torch.cat(
                    (rgb_predicted.detach(), normal_map.detach()), dim=-2)

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

            if it + 1 in report:
                evaluate_inversion(it + 1)

        t2 = time.time()
        idx += test_bs
        print(
            f'[{idx}/{len(image_indices)}] Finished batch in {t2-t1} s ({(t2-t1)/test_bs} s/img)'
        )

        if idx % 512 == 0:
            # Save report checkpoint
            with utils.open_file(report_checkpoint_path, 'wb') as f:
                torch.save({
                    'report': report,
                    'idx': idx,
                    'test_bs': test_bs,
                }, f)

    # Consolidate stats
    for report_entry in report.values():
        for k, v in list(report_entry.items()):
            if len(v) == 0:
                del report_entry[k]
            else:
                report_entry[k] = torch.cat(v, dim=0)

    print()
    print('Useful information:')
    print('psnr_random: PSNR evaluated on novel views (in the test set, if available)')
    print('ssim_random: SSIM evaluated on novel views (in the test set, if available)')
    print('rot_error: rotation error in degrees (only reliable on synthetic datasets)')
    print('fid_random: FID of selected split, evaluated against stats of train split')
    print('fid_random_test: FID of selected split, evaluated against stats of test split')
    print()
    report_str_full = ''
    for iter_num, report_entry in report.items():
        report_str = f'[{iter_num} iterations]'
        for elem in [
                'psnr', 'psnr_random', 'lpips', 'lpips_random', 'ssim',
                'ssim_random', 'iou', 'rot_error'
        ]:
            if elem in report_entry:
                elem_val = report_entry[elem].mean().item()
                report_str += f' {elem} {elem_val:.05f}'
                report_entry[f'{elem}_avg'] = elem_val
                writer.add_scalar(f'report/{elem}', elem_val, iter_num)

        def add_inception_report(report_entry_key, tensorboard_key):
            global report_str
            if report_entry_key not in report_entry:
                return None
            fid_stats = fid.calculate_stats(
                report_entry[report_entry_key].numpy())
            fid_value = fid.calculate_frechet_distance(
                *fid_stats, *train_eval_split.fid_stats)
            report_entry[tensorboard_key] = fid_value
            report_str += f' {tensorboard_key} {fid_value:.02f}'
            del report_entry[report_entry_key]
            writer.add_scalar(f'report/{tensorboard_key}', fid_value, iter_num)
            if use_testset:
                fid_test = fid.calculate_frechet_distance(
                    *fid_stats, *test_split.fid_stats)
                report_entry[tensorboard_key + '_test'] = fid_test
                report_str += f' {tensorboard_key}_test {fid_test:.02f}'
                writer.add_scalar(f'report/{tensorboard_key}_test', fid_test,
                                  iter_num)
            return fid_value

        add_inception_report('inception_activations_front', 'fid_front')
        fid_value = add_inception_report('inception_activations_random',
                                         'fid_random')

        print(report_str)
        report_str_full += report_str + '\n'

    report_file_in = os.path.join(report_dir_effective, 'report')
    with utils.open_file(report_file_in + '.pth', 'wb') as f:
        torch.save(report, f)
    with utils.open_file(report_file_in + '.txt', 'w') as f:
        f.write(args.resume_from + '\n')
        f.write(cfg_string + '\n')
        f.write(report_str_full)
