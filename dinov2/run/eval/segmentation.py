# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""mmseg 0.27 launcher for frozen-backbone semantic-segmentation probes."""

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, import_modules_from_strings
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes


def get_args_parser():
    parser = argparse.ArgumentParser("ADE20K linear segmentation")
    parser.add_argument("config", help="mmseg config file")
    parser.add_argument("--work-dir", help="directory for logs and checkpoints")
    parser.add_argument("--data-root", help="ADEChallengeData2016 directory")
    parser.add_argument(
        "--pretrained-weights",
        help="optional local/URL backbone checkpoint; otherwise use the timm pretrained weights",
    )
    parser.add_argument("--resume-from", help="segmentation checkpoint to resume")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=("none", "pytorch", "slurm", "mpi"), default="pytorch")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser


def _override_paths(cfg, args):
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir") is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    if args.data_root:
        for split in ("train", "val", "test"):
            cfg.data[split].data_root = args.data_root
    if args.pretrained_weights:
        cfg.model.backbone.pretrained = False
        cfg.model.backbone.checkpoint_path = args.pretrained_weights
    if args.resume_from:
        cfg.resume_from = args.resume_from


def main(args=None):
    args = get_args_parser().parse_args(args)
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    _override_paths(cfg, args)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    setup_multi_processes(cfg)

    if args.launcher == "none":
        distributed = False
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    root_logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    env_info = "\n".join(f"{key}: {value}" for key, value in collect_env().items())
    root_logger.info("Environment info:\n%s", env_info)
    root_logger.info("Distributed training: %s", distributed)
    root_logger.info("Config:\n%s", cfg.pretty_text)

    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta = dict(env_info=env_info, seed=seed, exp_name=osp.basename(args.config))

    model = build_segmentor(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    if not distributed:
        warnings.warn("SyncBN requires DDP; converting the linear head to regular BatchNorm")
        model = revert_sync_batchnorm(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
