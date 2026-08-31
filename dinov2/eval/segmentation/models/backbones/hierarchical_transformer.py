# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Mapping, Optional, Sequence

import torch
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES

from dinov2.models.hierarchical_transformer import TimmHierarchicalBackbone


@BACKBONES.register_module()
class TimmHierarchicalTransformer(BaseModule):
    """Frozen timm PVTv2/Swin backbone for mmseg 0.27 linear probing."""

    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        frozen: bool = True,
        model_kwargs: Optional[Mapping] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.frozen = frozen
        self.backbone = TimmHierarchicalBackbone(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            model_kwargs=model_kwargs,
        )
        if max(self.out_indices) >= self.backbone.num_stages:
            raise ValueError(f"out_indices={self.out_indices} exceed the {self.backbone.num_stages} available stages")
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def init_weights(self):
        # timm initializes (and optionally loads) the model in __init__.
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor):
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            features = self.backbone.model.forward_intermediates(
                images,
                indices=list(self.out_indices),
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        return tuple(features)
