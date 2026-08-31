# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Mapping, Optional
from urllib.parse import urlparse

import torch
from torch import nn


logger = logging.getLogger("dinov2")


def _unwrap_state_dict(checkpoint):
    """Extract a model state dict from common training-checkpoint layouts."""
    state_dict = checkpoint
    for key in ("teacher", "model", "state_dict"):
        if isinstance(state_dict, Mapping) and key in state_dict and isinstance(state_dict[key], Mapping):
            state_dict = state_dict[key]
            break
    if not isinstance(state_dict, Mapping):
        raise TypeError("The checkpoint does not contain a state dict")

    prefixes = ("module.", "backbone.", "model.")
    state_dict = dict(state_dict)
    for prefix in prefixes:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def load_timm_pretrained_weights(model: nn.Module, pretrained_weights: str):
    """Load either a timm checkpoint or a checkpoint saved by this adapter."""
    if urlparse(pretrained_weights).scheme:
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    message = model.load_state_dict(state_dict, strict=False)
    loaded_keys = len(state_dict) - len(message.unexpected_keys)
    if loaded_keys == 0:
        raise RuntimeError(f'No backbone weights from "{pretrained_weights}" matched the timm model')
    logger.info(
        'Loaded hierarchical backbone weights from "%s" (%d matched keys): %s',
        pretrained_weights,
        loaded_keys,
        message,
    )
    return message


class TimmHierarchicalBackbone(nn.Module):
    """Expose timm hierarchical ViTs through DINOv2's feature API.

    PVTv2 and Swin do not have a class token.  For ImageNet linear probing,
    the spatial mean of each selected stage is returned as its global token.
    The default probe therefore uses the final-stage global average, matching
    the native classifier readout of both model families.
    """

    is_hierarchical = True
    linear_probe_n_last_blocks = (1,)
    linear_probe_use_avgpool = (False,)

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Mapping] = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Hierarchical backbones require timm==1.0.19") from exc

        kwargs = dict(model_kwargs or {})
        kwargs.setdefault("num_classes", 0)
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained and checkpoint_path is None, **kwargs)
        if not hasattr(self.model, "forward_intermediates"):
            raise TypeError(f'timm model "{model_name}" has no forward_intermediates(); use timm==1.0.19')
        if checkpoint_path:
            load_timm_pretrained_weights(self.model, checkpoint_path)

        self.num_stages = len(self.model.feature_info)
        self.embed_dim = self.model.num_features
        self.num_features = self.model.num_features

    def forward_features(self, images: torch.Tensor):
        return self.model.forward_features(images)

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        n: int = 1,
        return_class_token: bool = False,
        **_kwargs,
    ):
        if not 1 <= n <= self.num_stages:
            raise ValueError(f"Requested {n} stages, but {self.model_name} exposes {self.num_stages}")
        features = self.model.forward_intermediates(
            images,
            indices=n,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        outputs = []
        for feature in features:
            patch_tokens = feature.flatten(2).transpose(1, 2)
            global_token = feature.mean(dim=(-2, -1))
            outputs.append((patch_tokens, global_token) if return_class_token else patch_tokens)
        return outputs

    def forward(self, images: torch.Tensor):
        feature = self.get_intermediate_layers(images, n=1, return_class_token=True)[0][1]
        return feature


def build_timm_hierarchical_backbone(args, pretrained_weights: Optional[str] = None):
    model_name = args.timm_model_name
    model_kwargs = dict(getattr(args, "timm_model_kwargs", {}) or {})
    pretrained = bool(getattr(args, "timm_pretrained", False))
    return TimmHierarchicalBackbone(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=pretrained_weights,
        model_kwargs=model_kwargs,
    )
