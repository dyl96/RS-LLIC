import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import Elic2022Official, LLIC
from torch.utils.tensorboard import SummaryWriter   
import os
import time

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    print("inter:", len(inter_params), "union:", len(union_params))
    print("all:", len(params_dict.keys()))

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


# def train_one_epoch(
#     model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
# ):
#     model.train()
#     device = next(model.parameters()).device

#     for i, d in enumerate(train_dataloader):
#         d = d.to(device)
#         optimizer.zero_grad()
#         aux_optimizer.zero_grad()

#         out_net = model(d)

#         out_criterion = criterion(out_net, d)
#         out_criterion["loss"].backward()
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
#         optimizer.step()

#         aux_loss = model.aux_loss()
#         aux_loss.backward()
#         aux_optimizer.step()

#         if i % 50 == 0:
#             if type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{i*len(d)}/{len(train_dataloader.dataset)}"
#                     f" ({100. * i / len(train_dataloader):.0f}%)]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {aux_loss.item():.2f}"
#                 )
#             else:
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{i*len(d)}/{len(train_dataloader.dataset)}"
#                     f" ({100. * i / len(train_dataloader):.0f}%)]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {aux_loss.item():.2f}"
#                 )


def train_one_epoch(
    teacher_model, model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse', alpha=1.0, beta=1.0):

    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        # 蒸馏损失计算 默认用mse
        with torch.no_grad():
            teacher_feats = extract_multi_scale(teacher_model, d)
        student_feats = extract_multi_scale(model, d)

        distill_loss = 0.0
        for t_feat, s_feat in zip(teacher_feats, student_feats):
            distill_loss += F.mse_loss(s_feat, t_feat.detach())

        # 压缩损失与蒸馏损失加权
        out_criterion = criterion(out_net, d)
        total_loss = beta * out_criterion["loss"] + alpha * distill_loss
        total_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 50 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {total_loss.item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f'\tDistill loss: {distill_loss.item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {total_loss.item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def freeze_module(m):
    for p in m.parameters():
        p.requires_grad = False


def extract_multi_scale_from_seq(seq, x):
    out = x
    feats = []
    prev_h, prev_w = out.shape[-2], out.shape[-1]
    for m in seq:
        out = m(out)
        h, w = out.shape[-2], out.shape[-1]
        if h < prev_h or w < prev_w:
            feats.append(out)
        prev_h, prev_w = h, w
    return feats

def extract_multi_scale(model, x):
    if hasattr(model, "extract_multi_scale"):
        return model.extract_multi_scale(x)
    if hasattr(model, "g_a") and isinstance(model.g_a, nn.Sequential):
        return extract_multi_scale_from_seq(model.g_a, x)
    y = model.g_a(x)
    return y

def save_checkpoint(state, is_best, epoch, save_path, filename):
    # torch.save(state, save_path + "checkpoint_latest.pth.tar")
    # if epoch % 5 == 0:
    #     torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint of teacher model")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--M", type=int, default=320,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="KD loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="compression loss weight")
    parser.add_argument("--freeze", action="store_true", default=False, help="weather to freeze the parameters")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    teacher_net = Elic2022Official(N=args.N, M=args.M)
    teacher_net = teacher_net.to(device)
    teacher_net.eval()

    if args.cuda and torch.cuda.device_count() > 1:
        teacher_net = CustomDataParallel(teacher_net)

    last_epoch = 0
    if args.checkpoint:  # load teacher checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        teacher_net.load_state_dict(checkpoint["state_dict"])
    

    net = LLIC(N=args.N, M=args.M)
    net = net.to(device)
    net.latent_codec = teacher_net.latent_codec
    net.g_s = teacher_net.g_s
    if args.freeze:
        freeze_module(net.latent_codec)
        freeze_module(net.g_s)
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)


    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        t1 = time.time()
        train_one_epoch(
            teacher_net,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type,
            alpha=args.alpha,
            beta=args.beta
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )
        t2 = time.time()
        print(f"Epoch {epoch} finished in {t2 - t1:.2f} seconds. Estimated time left: {(args.epochs - epoch - 1) * (t2 - t1) / 3600:.2f} hours.")


if __name__ == "__main__":
    main(sys.argv[1:])