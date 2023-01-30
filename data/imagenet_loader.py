import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from data import datasets


def load_dataset(cfg, device):
    dataset_config = get_dataset_config(cfg)
    train_split, train_eval_split, test_split = load_custom(cfg, dataset_config, device)

    return dataset_config, train_split, train_eval_split, test_split

def get_dataset_config(cfg):
    dataset_config = cfg.dataset_config
    return dataset_config

def load_custom(cfg, dataset_config, device):

    train_split = DatasetSplit(device)
    train_eval_split = DatasetSplit(device)
    test_split = None

    all_images = []
    all_images_highres = [] if dataset_config.augment_p > 0 else None
    all_images_fid = []
    all_poses = []
    all_focal = []
    all_bbox = []
    all_classes_id = []
    all_classes = []

    all_poses_fid = []
    all_focal_fid = []
    all_bbox_fid = []
    all_classes_id_fid = []
    all_classes_fid = []

    for imagenet_cls in cfg.dataset:
        dataset_inst = lambda *fn_args, **fn_kwargs: datasets.CustomDataset(
            imagenet_cls, *fn_args, **fn_kwargs, root_dir=cfg.data_path
        )

        img_size = dataset_config.resolution
        img_size_train = img_size * 2 if dataset_config.augment_p > 0 else img_size
        dataset = dataset_inst('train',
                            img_size=img_size_train,
                            crop=False,
                            add_mirrored=True)
        dataset_fid = dataset_inst('train',
                                img_size=img_size,
                                crop=True,
                                add_mirrored=False)
        loader = torch.utils.data.DataLoader(dataset,
                                            shuffle=False,
                                            batch_size=32,
                                            num_workers=8,
                                            pin_memory=False)
        loader_fid = torch.utils.data.DataLoader(dataset_fid,
                                                shuffle=False,
                                                batch_size=32,
                                                num_workers=8,
                                                pin_memory=False)

        for _, sample in enumerate(tqdm(loader)):
            if dataset_config.augment_p > 0:
                all_images_highres.append(sample['img'].clamp(-1, 1).permute(
                    0, 2, 3, 1))
                all_images.append(
                    F.avg_pool2d(sample['img'], 2).clamp(-1, 1).permute(0, 2, 3, 1))
            else:
                all_images.append(sample['img'].clamp(-1, 1).permute(0, 2, 3, 1))
            all_poses.append(sample['pose'])
            all_focal.append(sample['focal'])
            all_bbox.append(sample['normalized_bbox'])
            all_classes_id.append(sample['class_id'])
            all_classes.append(sample['class'])

        for _, sample in enumerate(tqdm(loader_fid)):
            all_images_fid.append(sample['img'].clamp(-1, 1).permute(0, 2, 3, 1))
            all_poses_fid.append(sample['pose'])
            all_focal_fid.append(sample['focal'])
            all_bbox_fid.append(sample['normalized_bbox'])
            all_classes_id_fid.append(sample['class_id'])
            all_classes_fid.append(sample['class'])

    train_split.images = torch.cat(all_images, dim=0)
    train_eval_split.images = torch.cat(all_images_fid, dim=0)
    all_images = None  # Free up memory
    all_images_fid = None
    if dataset_config.augment_p > 0:
        train_split.images_highres = torch.cat(all_images_highres, dim=0)
        all_images_highres = None  # Free up memory
    train_split.tform_cam2world = torch.cat(all_poses, dim=0)
    train_split.focal_length = torch.cat(all_focal, dim=0).squeeze(1)
    train_split.bbox = torch.cat(all_bbox, dim=0)
    train_split.classes_id = torch.cat(all_classes_id, dim=0)
    train_split.classes = np.concatenate(all_classes, axis=0)
    train_split.num_classes = train_split.classes_id.max().item() + 1
    for id_, class_ in zip(train_split.classes_id, train_split.classes):
        if class_ not in train_split.class_to_id:
            train_split.class_to_id[class_] = id_
            train_split.id_to_class[id_] = class_

    train_eval_split.tform_cam2world = torch.cat(all_poses_fid, dim=0)
    train_eval_split.focal_length = torch.cat(all_focal_fid, dim=0).squeeze(1)
    train_eval_split.bbox = torch.cat(all_bbox_fid, dim=0)
    train_eval_split.classes_id = torch.cat(all_classes_id_fid, dim=0)
    train_eval_split.classes = np.concatenate(all_classes_fid, axis=0)
    train_eval_split.num_classes = train_split.num_classes
    train_eval_split.class_to_id = train_split.class_to_id
    train_eval_split.id_to_class = train_eval_split.id_to_class

    return train_split, train_eval_split, test_split

class DatasetSplitView:

    def __init__(self, parent, idx):
        self.parent = parent
        self.idx = idx

    def __getattr__(self, attr):
        if isinstance(attr, (list, tuple)):
            outputs = []
            for elem in attr:
                parent_attr = getattr(self.parent, elem)
                if parent_attr is None:
                    outputs.append(None)
                else:
                    outputs.append(parent_attr[self.idx].to(parent.device))
            return outputs
        else:
            parent_attr = getattr(self.parent, attr)
            if parent_attr is None:
                return None
            else:
                return parent_attr[self.idx].to(self.parent.device)


class DatasetSplit:

    def __init__(self, device='cpu'):
        self.device = device
        self.images = None
        self.images_highres = None
        self.tform_cam2world = None
        self.focal_length = None
        self.bbox = None
        self.center = None
        self.classes_id = None
        self.classes = None
        self.num_classes = None
        self.class_to_id = {}
        self.id_to_class = {}

        self.fid_stats = None
        self.eval_indices = None
        self.eval_indices_perm = None

    def __getitem__(self, idx):
        return DatasetSplitView(self, idx)
