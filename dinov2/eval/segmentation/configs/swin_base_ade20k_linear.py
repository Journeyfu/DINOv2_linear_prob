# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

_base_ = ["./ade20k_linear_base.py"]

model = dict(
    backbone=dict(
        type="TimmHierarchicalTransformer",
        model_name="swin_base_patch4_window7_224.ms_in1k",
        out_indices=(0, 1, 2, 3),
        pretrained=True,
        frozen=True,
        model_kwargs=dict(img_size=512, strict_img_size=False),
    ),
    decode_head=dict(in_channels=[1024], channels=1024),
)

work_dir = "./work_dirs/swin_base_ade20k_linear"
