# regions_test

A worked example of running [michalsr/regions](https://github.com/michalsr/regions)'
region-feature pipeline (`region_features/` in [../regions](../regions)) on one image from
the NVRD dataset, `shape_16` (a "novel" object).

## Pipeline

1. **Locate the image** in the HF dataset without downloading all ~19K images:
   ```bash
   python find_shape16.py shape_16      # scans 971 parquet shards' metadata columns only (~20s)
   python fetch_image.py shape_16       # downloads just that image -> data/novel/shape_16.png
   ```
2. **Generate SAM regions** (from `../regions/region_features/generate_regions.py`):
   ```bash
   python ../regions/region_features/generate_regions.py \
     --input data/novel --output output/sam_masks \
     --checkpoint <path to sam_vit_b_01ec64.pth> --model-type vit_b --convert-to-rle
   ```
   Checkpoint: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (375MB, not checked in).
3. **Extract DINOv2 ViT-L/14 features** (`../regions/region_features/extract_features.py`, downloads the model via `torch.hub` on first run):
   ```bash
   python ../regions/region_features/extract_features.py \
     --image_dir data/novel --feature_dir output/features
   ```
4. **Pool features within each region** (`../regions/region_features/process_regions.py`):
   ```bash
   python ../regions/region_features/process_regions.py \
     --feature_dir output/features --mask_dir output/sam_masks \
     --region_feature_dir output/region_features \
     --interpolate upsample --pooling_method average --dtype bf16
   ```
5. **Visualize**:
   ```bash
   python visualize.py shape_16   # -> output/shape_16_regions_viz.png
   ```

## Results (checked in)

- `data/novel/shape_16.png` — input image
- `output/sam_masks/shape_16.json` — 8 SAM regions (COCO RLE)
- `output/features/shape_16.pkl` — DINOv2 ViT-L/14 feature map, shape `(1, 1024, 74, 74)`
- `output/region_features/shape_16.pkl` — 8 pooled 1024-d region features
- `../shape_16_regions_viz.png` — visualization (original / masks-on-image / masks-only)

SAM (ViT-B checkpoint, default settings) split `shape_16` into 8 regions matching its visible
parts: the cube "head", the arched handle, the leg/base, the small connector joint, plus
background.

## Environment

Not included in this repo (see main [requirements.txt](../requirements.txt) plus, for this
pipeline specifically): `torch`, `torchvision`, `segment-anything`, `pycocotools`,
`opencv-python-headless`, `timm`, `openai-clip` (`pip install git+https://github.com/openai/CLIP.git`),
`pyyaml`, `coloredlogs`, `fsspec[http]`. Tested with `torch==2.2.2+cu118` /
`transformers==4.44.2` (newer transformers requires `torch>=2.5`).
