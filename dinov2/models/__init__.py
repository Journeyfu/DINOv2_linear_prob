# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from .hierarchical_transformer import build_timm_hierarchical_backbone


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224, pretrained_weights=None):
    args.arch = args.arch.removesuffix("_memeff")
    if args.arch == "timm_hierarchical":
        teacher = build_timm_hierarchical_backbone(args, pretrained_weights=pretrained_weights)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = build_timm_hierarchical_backbone(args, pretrained_weights=pretrained_weights)
        embed_dim = student.embed_dim
    elif "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            in_chans=args.in_chans,
            channel_adaptive=args.channel_adaptive,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False, pretrained_weights=None):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
        pretrained_weights=pretrained_weights,
    )
