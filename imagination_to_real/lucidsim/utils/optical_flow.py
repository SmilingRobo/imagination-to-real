from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchtyping import TensorType


def warp(x, flo):
    """
    From RAFT: https://github.com/princeton-vl/RAFT/issues/64

    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow (1 -> 2)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size())
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.to("cuda")
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def warp_forward(x, flo, return_mask=False):
    """
    warp an image/tensor (im1) to im2, according to the optical flow
    x: [B, C, H, W] (im1)
    flo: [B, 2, H, W] flow (1 -> 2)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    mask = torch.ones(x.size())

    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.to("cuda")
    # Invert the flow by multiplying by -1
    vgrid = grid - flo  # Change here
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    if return_mask:
        return output, mask
    return output


def plot(imgs, **imshow_kwargs):
    import matplotlib.pyplot as plt

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            # img = F2.to_pil_image(img.to("cpu"))
            img = img.permute(1, 2, 0).cpu()

            if img.dtype == torch.float32:
                img = (img + 1) / 2

            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


def preprocess(
        batch: TensorType["batch", "channel", "height", "width"],
) -> TensorType["batch", "channel", "resize_height", "resize_width", torch.float32]:
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=RunArgs.resize_to),
        ]
    )
    batch = transforms(batch)
    return batch


def load_images(images: List[str], root: Path) -> TensorType["batch", "channel", "height", "width"]:
    images = np.concatenate([np.array(Image.open(root / image))[None, ..., :3] for image in images])
    frames = torch.from_numpy(images).permute(0, 3, 1, 2)
    return frames


def compute_consistency_mask(target, predicted, threshold=0.2):
    delta = torch.norm(target - predicted, dim=1)
    consistency_mask = torch.abs(delta < threshold)
    return consistency_mask.float()
