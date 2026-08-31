# PVTv2-B2 / Swin-Base linear probing

This extension keeps the backbone frozen and reuses the DINOv2 evaluation
protocols for ImageNet-1K classification and ADE20K semantic segmentation.
The supported timm checkpoints are:

- `pvt_v2_b2.in1k`
- `swin_base_patch4_window7_224.ms_in1k`

Both are hierarchical transformers without a class token.  ImageNet probing
therefore uses the native readout of these architectures: global average
pooling of the final stage followed by one linear layer.  The DINOv2 training
transform, 10 epochs, SGD, cosine schedule, zero weight decay, batch-size LR
scaling, and 13-value LR sweep are unchanged.  `--n-last-blocks 1 4
--pooling-modes global global+patch` can be used to explicitly run the full
DINOv2 feature grid over the four hierarchy stages.

## Installation

Use the normal environment for ImageNet and the extras environment for
ADE20K.  `timm==1.0.19` is included in both environment definitions.

```bash
conda env create -f conda.yaml
# or, for ADE20K:
conda env create -f conda-extras.yaml
```

## ImageNet-1K

First generate the DINOv2 ImageNet metadata as described in the main README.
Then launch one process per GPU. `ROOT` is the directory containing `train/`
and `val/`; `EXTRA` contains the generated `.npy` metadata.

```bash
torchrun --nproc-per-node=8 dinov2/eval/linear.py \
  --config-file dinov2/configs/eval/pvtv2_b2_in1k.yaml \
  --output-dir work_dirs/pvtv2_b2_imagenet_linear \
  --train-dataset ImageNet:split=TRAIN:root=ROOT:extra=EXTRA \
  --val-dataset ImageNet:split=VAL:root=ROOT:extra=EXTRA

torchrun --nproc-per-node=8 dinov2/eval/linear.py \
  --config-file dinov2/configs/eval/swin_base_in1k.yaml \
  --output-dir work_dirs/swin_base_imagenet_linear \
  --train-dataset ImageNet:split=TRAIN:root=ROOT:extra=EXTRA \
  --val-dataset ImageNet:split=VAL:root=ROOT:extra=EXTRA
```

The configs use timm's published ImageNet-1K weights. To evaluate a local
backbone checkpoint, add `--pretrained-weights /path/to/backbone.pth`; this
disables timm's automatic download for that run. Common `model`, `teacher`,
`state_dict`, `module.`, `backbone.`, and `model.` checkpoint wrappers are
handled automatically.

## ADE20K

The expected dataset layout is:

```text
ADEChallengeData2016/
  images/{training,validation}/
  annotations/{training,validation}/
```

Launch with DDP:

```bash
torchrun --nproc-per-node=8 dinov2/run/eval/segmentation.py \
  dinov2/eval/segmentation/configs/pvtv2_b2_ade20k_linear.py \
  --data-root /path/to/ADEChallengeData2016 \
  --work-dir work_dirs/pvtv2_b2_ade20k_linear

torchrun --nproc-per-node=8 dinov2/run/eval/segmentation.py \
  dinov2/eval/segmentation/configs/swin_base_ade20k_linear.py \
  --data-root /path/to/ADEChallengeData2016 \
  --work-dir work_dirs/swin_base_ade20k_linear
```

The ADE20K protocol is the one used by the DINOv2-linked DVT evaluation:
512x512 crops, batch size 2 per GPU, 40k iterations, AdamW at 1e-3 with
weight decay 1e-4, 1.5k linear warmup, polynomial decay, and sliding-window
validation.  Only the final hierarchy stage is consumed by a trainable SyncBN
plus 1x1 classifier; all backbone parameters and backbone normalization
statistics stay frozen. A local backbone can again be selected with
`--pretrained-weights /path/to/backbone.pth`.

Protocol references: [test-time-registers evaluation instructions](https://github.com/nickjiang2378/test-time-registers#evaluation)
and the [DVT ADE20K linear config](https://github.com/Jiawei-Yang/Denoising-ViT/blob/main/evaluation/configs/vitb_ade20k_linear_config.py).
