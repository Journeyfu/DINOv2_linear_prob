# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

_base_ = ["./ade20k_linear_base.py"]

model = dict(
    backbone=dict(
        type="TimmHierarchicalTransformer",
        model_name="pvt_v2_b2.in1k",
        out_indices=(0, 1, 2, 3),
        pretrained=True,
        frozen=True,
    ),
    decode_head=dict(in_channels=[512], channels=512),
)

work_dir = "./work_dirs/pvtv2_b2_ade20k_linear"
