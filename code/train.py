import os
import os.path as osp
import time
import math
import json
import random
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from utils import increment_path

import wandb
import numpy as np


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--seed", type=int, default=4669)
    parser.add_argument("--data_dir", type=str, default="../data/medical")
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--ignore_tags", type=list, default=["masked", "excluded-region", "maintable", "stamp"])
    parser.add_argument("--wandb_name", type=str, default="default_run_name")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def do_training(config, seed, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epochs, save_interval, ignore_tags, wandb_name):
    seed_everything(seed)  # set seed
    
    # checkpoint directory initialization
    model_dir = increment_path(os.path.join(model_dir, wandb_name))
    model_name = model_dir.split("/")[-1]
    
    # logging with config.json
    with open(os.path.join(model_dir, f"{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)
    
    # logging with wandb
    wandb.init(project="level2_data_centric",
               name=model_name,
               config=config)
    
    dataset = SceneTextDataset(
        data_dir,
        split="train",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epochs // 2], gamma=0.1)
    
    model.train()
    for epoch in range(max_epochs):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))
                
                img = img.to(non_blocking=True)
                gt_score_map = gt_score_map.to(non_blocking=True)
                gt_geo_map = gt_geo_map.to(non_blocking=True)
                roi_mask = roi_mask.to(non_blocking=True)
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"]
                }
                pbar.set_postfix(val_dict)
                
                # logging with wandb
                wandb.log({
                    "Cls loss": extra_info['cls_loss'],
                    "Angle loss": extra_info['angle_loss'],
                    "IoU loss": extra_info['iou_loss'],
                    "loss": loss_val
                })

        scheduler.step()

        print("Mean loss: {:.4f} | Elapsed time: {}".format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(args, **args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    print(args)  # CLI에 configs 표시
    main(args)
