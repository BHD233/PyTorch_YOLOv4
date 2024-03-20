import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch

# from models.yolo import Model
from models.models import *
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import (
    labels_to_class_weights,
    increment_path,
    labels_to_image_weights,
    init_seeds,
    fitness,
    fitness_p,
    fitness_r,
    fitness_ap50,
    fitness_ap,
    fitness_f,
    strip_optimizer,
    get_latest_run,
    check_dataset,
    check_file,
    check_git_status,
    check_img_size,
    print_mutation,
    set_logging,
)
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import (
    ModelEMA,
    select_device,
    intersect_dicts,
    torch_distributed_zero_first,
)

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = (
            (bboxes_a[:, None, 0] + bboxes_a[:, None, 2])
            - (bboxes_b[:, 0] + bboxes_b[:, 2])
        ) ** 2 / 4 + (
            (bboxes_a[:, None, 1] + bboxes_a[:, None, 3])
            - (bboxes_b[:, 1] + bboxes_b[:, 3])
        ) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # intersection bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        con_br = torch.max(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2
                )
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=8):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 640
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.batch = batch

        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        (
            self.masked_anchors,
            self.ref_anchors,
            self.grid_x,
            self.grid_y,
            self.anchor_w,
            self.anchor_h,
        ) = ([], [], [], [], [], [])

        for i in range(3):
            all_anchors_grid = [
                (w / self.strides[i], h / self.strides[i]) for w, h in self.anchors
            ]
            masked_anchors = np.array(
                [all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32
            )
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .to(device)
            )
            grid_y = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def convert_labels(self, labels):
        if len(labels.shape) > 2:
            return labels
        a = []
        cur = []
        for prev_label in range(self.batch):
            for label in labels:
                if prev_label == label[0]:
                    cur.append([label[1], label[2], label[3], label[4], label[0]])

            for i in range(len(cur), 40):
                cur.append([0, 0, 0, 0, 0])

            a.append(cur)
        return torch.Tensor(a)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(
            batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes
        ).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(
            device=self.device
        )
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(
            self.device
        )
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(
            self.device
        )

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (
            self.strides[output_id] * 2
        )
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (
            self.strides[output_id] * 2
        )
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(), self.ref_anchors[output_id], CIoU=True
            )

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.anch_masks[output_id][0])
                | (best_n_all == self.anch_masks[output_id][1])
                | (best_n_all == self.anch_masks[output_id][2])
            )

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.reshape(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0]
                        + 1e-16
                    )
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1]
                        + 1e-16
                    )
                    target[b, a, j, i, 4] = 1
                    target[
                        b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()
                    ] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                    )
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.reshape(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]]
            )

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]
            labels = self.convert_labels(labels)
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id
            )
            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            import pickle

            with open("output.pickle", "wb") as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("output.pickle", "wb") as handle:
                pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)

            iou = bbox_iou(output[..., :4].T, target[..., :4], x1y1x2y2=True, CIoU=True)

            loss_obj += F.binary_cross_entropy(
                input=output[..., 4], target=target[..., 4], reduction="sum"
            )
            loss_cls += F.binary_cross_entropy(
                input=output[..., 5:], target=target[..., 5:], reduction="sum"
            )
            loss_l2 += F.mse_loss(input=output, target=target, reduction="sum")

        loss = (1.0 - iou).mean() + loss_obj + loss_cls

        return loss


def bbox_iou(
    box1,
    box2,
    x1y1x2y2=True,
    GIoU=False,
    DIoU=False,
    CIoU=False,
    EIoU=False,
    ECIoU=False,
    eps=1e-9,
):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU or ECIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if (
            CIoU or DIoU or EIoU or ECIoU
        ):  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:  # Efficient IoU https://arxiv.org/abs/2101.08158
                rho3 = (w1 - w2) ** 2
                c3 = cw**2 + eps
                rho4 = (h1 - h2) ** 2
                c4 = ch**2 + eps
                return iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU
            elif ECIoU:
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                rho3 = (w1 - w2) ** 2
                c3 = cw**2 + eps
                rho4 = (h1 - h2) ** 2
                c4 = ch**2 + eps
                return iou - v * alpha - rho2 / c2 - rho3 / c3 - rho4 / c4  # ECIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


criterion = Yolo_loss(
    device="cpu",
    batch=8,
    n_classes=22,
)


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f"Hyperparameters {hyp}")
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    print(weights)
    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )  # check

    # Model
    pretrained = weights.endswith(".pt")
    print(weights, pretrained)
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Darknet("yolov4.conv.137.pth", n_classes=22).to(device)  # create
        state_dict = {
            k: v
            for k, v in ckpt["model"].items()
            if model.state_dict()[k].numel() == v.numel()
        }
        model.load_state_dict(state_dict, strict=False)
        print(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )  # report
    else:
        model = Darknet("yolov4.conv.137.pth", n_classes=22).to(device)  # create

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if ".bias" in k:
            pg2.append(v)  # biases
        elif "Conv2d.weight" in k:
            pg1.append(v)  # apply weight_decay
        elif "m.weight" in k:
            pg1.append(v)  # apply weight_decay
        elif "w.weight" in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(
            pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": hyp["weight_decay"]}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    logger.info(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"])
        + hyp["lrf"]
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    if wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project="YOLOv4" if opt.project == "runs/train" else Path(opt.project).stem,
            name=save_dir.stem,
            id=ckpt.get("wandb_id") if "ckpt" in locals() else None,
        )

    # Resume
    start_epoch, best_fitness = 0, 0.0
    (
        best_fitness_p,
        best_fitness_r,
        best_fitness_ap50,
        best_fitness_ap,
        best_fitness_f,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0)
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
            best_fitness_p = ckpt["best_fitness_p"]
            best_fitness_r = ckpt["best_fitness_r"]
            best_fitness_ap50 = ckpt["best_fitness_ap50"]
            best_fitness_ap = ckpt["best_fitness_ap"]
            best_fitness_f = ckpt["best_fitness_f"]

        # Results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert (
                start_epoch > 0
            ), "%s training to %g epochs is finished, nothing to resume." % (
                weights,
                epochs,
            )
        if epochs < start_epoch:
            logger.info(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = 64  # int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.workers,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert (
        mlc < nc
    ), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
        mlc,
        nc,
        opt.data,
        nc - 1,
    )

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            batch_size * 2,
            gs,
            opt,
            hyp=hyp,
            cache=opt.cache_images and not opt.notest,
            rect=True,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
        )[
            0
        ]  # testloader

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir=save_dir)
                # if tb_writer:
                #     tb_writer.add_histogram("classes", c, 0)
                if wandb:
                    wandb.log(
                        {
                            "Labels": [
                                wandb.Image(str(x), caption=x.name)
                                for x in save_dir.glob("*labels*.png")
                            ]
                        }
                    )

            # Anchors
            # if not opt.noautoanchor:
            #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(
        device
    )  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(
        round(hyp["warmup_epochs"] * nb), 1000
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info(
        "Image sizes %g train, %g test\n"
        "Using %g dataloader workers\nLogging results to %s\n"
        "Starting training for %g epochs..."
        % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs)
    )

    torch.save(model, wdir / "init.pt")

    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = (
                    model.class_weights.cpu().numpy() * (1 - maps) ** 2
                )  # class weights
                iw = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=cw
                )  # image weights
                dataset.indices = random.choices(
                    range(dataset.n), weights=iw, k=dataset.n
                )  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (
                    torch.tensor(dataset.indices)
                    if rank == 0
                    else torch.zeros(dataset.n)
                ).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (
            imgs,
            targets,
            paths,
            _,
        ) in (
            pbar
        ):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            hyp["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                        )

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )
            import copy

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss = criterion(
                    pred, copy.deepcopy(targets)
                )  # loss scaled by batch_size
                loss_items = loss  # loss scaled by batch_size
                if rank != -1:
                    loss *= (
                        opt.world_size
                    )  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 6) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f"train_batch{ni}.jpg"  # filename
                    plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log(
                        {
                            "Mosaics": [
                                wandb.Image(str(x), caption=x.name)
                                for x in save_dir.glob("train*.jpg")
                            ]
                        }
                    )

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                if epoch >= 3:
                    results, maps, times = test.test(
                        opt.data,
                        batch_size=batch_size * 2,
                        imgsz=imgsz_test,
                        model=ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                        single_cls=opt.single_cls,
                        dataloader=testloader,
                        save_dir=save_dir,
                        plots=plots and final_epoch,
                        log_imgs=opt.log_imgs if wandb else 0,
                    )

            # Write
            with open(results_file, "a") as f:
                f.write(
                    s + "%10.4g" * 7 % results + "\n"
                )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system(
                    "gsutil cp %s gs://%s/results/results%s.txt"
                    % (results_file, opt.bucket, opt.name)
                )

            # Log
            tags = [
                "train/box_loss",
                "train/obj_loss",
                "train/cls_loss",  # train loss
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/mAP_0.5:0.95",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",  # val loss
                "x/lr0",
                "x/lr1",
                "x/lr2",
            ]  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                # if tb_writer:
                #     tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_p = fitness_p(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_r = fitness_r(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap50 = fitness_ap50(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap = fitness_ap(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f = fitness_f(
                    np.array(results).reshape(1, -1)
                )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            else:
                fi_f = 0.0
            if fi > best_fitness:
                best_fitness = fi
            if fi_p > best_fitness_p:
                best_fitness_p = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r = fi_r
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap = fi_ap
            if fi_f > best_fitness_f:
                best_fitness_f = fi_f

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, "r") as f:  # create checkpoint
                    ckpt = {
                        "epoch": epoch,
                        "best_fitness": best_fitness,
                        "best_fitness_p": best_fitness_p,
                        "best_fitness_r": best_fitness_r,
                        "best_fitness_ap50": best_fitness_ap50,
                        "best_fitness_ap": best_fitness_ap,
                        "best_fitness_f": best_fitness_f,
                        "training_results": f.read(),
                        "model": (
                            ema.ema.module.state_dict()
                            if hasattr(ema, "module")
                            else ema.ema.state_dict()
                        ),
                        "optimizer": None if final_epoch else optimizer.state_dict(),
                        "wandb_id": wandb_run.id if wandb else None,
                    }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
                if best_fitness == fi:
                    torch.save(ckpt, wdir / "best_overall.pt")
                if best_fitness_p == fi_p:
                    torch.save(ckpt, wdir / "best_p.pt")
                if best_fitness_r == fi_r:
                    torch.save(ckpt, wdir / "best_r.pt")
                if best_fitness_ap50 == fi_ap50:
                    torch.save(ckpt, wdir / "best_ap50.pt")
                if best_fitness_ap == fi_ap:
                    torch.save(ckpt, wdir / "best_ap.pt")
                if best_fitness_f == fi_f:
                    torch.save(ckpt, wdir / "best_f.pt")
                if epoch == 0:
                    torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                if ((epoch + 1) % 25) == 0:
                    torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                if epoch >= (epochs - 5):
                    torch.save(ckpt, wdir / "last_{:03d}.pt".format(epoch))
                elif epoch >= 420:
                    torch.save(ckpt, wdir / "last_{:03d}.pt".format(epoch))
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = opt.name if opt.name.isnumeric() else ""
        fresults, flast, fbest = (
            save_dir / f"results{n}.txt",
            wdir / f"last{n}.pt",
            wdir / f"best{n}.pt",
        )
        for f1, f2 in zip(
            [wdir / "last.pt", wdir / "best.pt", results_file], [flast, fbest, fresults]
        ):
            if f1.exists():
                os.rename(f1, f2)  # rename
                if str(f2).endswith(".pt"):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    (
                        os.system("gsutil cp %s gs://%s/weights" % (f2, opt.bucket))
                        if opt.bucket
                        else None
                    )  # upload
        # Finish
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                wandb.log(
                    {
                        "Results": [
                            wandb.Image(str(save_dir / x), caption=x)
                            for x in ["results.png", "precision-recall_curve.png"]
                        ]
                    }
                )
        logger.info(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )
    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov4.weights", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="data/coco.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp", type=str, default="data/hyp.scratch.yaml", help="hyperparameters path"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--log-imgs",
        type=int,
        default=16,
        help="number of images for W&B logging, max 100",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    print(opt.cfg)
    if opt.resume:  # resume an interrupted run
        ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run()
        )  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "opt.yaml") as f:
            print("xam", Path(ckpt).parent.parent / "opt.yaml")
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.weights, opt.resume = ckpt, True
        logger.info("Resuming training from %s" % ckpt)
        print(opt.cfg)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = (
            check_file(opt.data),
            check_file(opt.cfg),
            check_file(opt.hyp),
        )  # check files
        assert len(opt.cfg) or len(
            opt.weights
        ), "either --cfg or --weights must be specified"
        opt.img_size.extend(
            [opt.img_size[-1]] * (2 - len(opt.img_size))
        )  # extend to 2 sizes (train, test)
        opt.name = "evolve" if opt.evolve else opt.name
        opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve
        )  # increment run
    print(opt.cfg)
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )  # distributed backend
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (opt.hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(
                f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/'
            )
            # tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, wandb)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (
                0,
                0.0,
                2.0,
            ),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (
                0,
                0.0,
                0.001,
            ),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
            "mixup": (1, 0.0, 1.0),
        }  # image mixup (probability)

        assert opt.local_rank == -1, "DDP mode not implemented for --evolve"
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / "hyp_evolved.yaml"  # save best result here
        if opt.bucket:
            os.system(
                "gsutil cp gs://%s/evolve.txt ." % opt.bucket
            )  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path(
                "evolve.txt"
            ).exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (
                        g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1
                    ).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}"
        )
