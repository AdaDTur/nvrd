"""
Run the released Aff-Grasp / GAT (Geometry-guided Affordance Transformer) checkpoint on our NVRD
"known" objects and visualize its per-pixel affordance predictions, for direct comparison against
our own VLM-based per-part labeling pipeline.

GAT's own vocabulary (AFF_LIST in the Aff-Grasp repo) is 8 UMD-style classes: grasp, cut, scoop,
pound, support, screw, contain, stick -- close to but not identical to our own UMD_GLOSSES (which
has "wrap-grasp" instead of "screw"/"stick").

Requires: Aff-Grasp repo cloned at ../Aff-Grasp (sibling of this repo), with:
  - affordance-learning/dinov2_vitb14_pretrain.pth (base DINOv2 ViT-B/14 weights)
  - the trained checkpoint (best_8_iou_68.4.pth from HF Gen1113/Model_for_Aff-Grasp)
  - models/dino/ops patched to use the pure-PyTorch deformable-attention fallback (no compiled
    CUDA extension available in this environment -- see the patch comments in
    models/dino/ops/functions/ms_deform_attn_func.py and modules/ms_deform_attn.py)
Also requires a depth-map render per object (regions_test/data/known_depth/<object>.png, from this
project's depth maps -- same Depth-Anything model family GAT itself uses for its own
pseudo-depth maps, so this is a fair, consistent input).

Usage (run from anywhere; internally chdir's to the Aff-Grasp checkout for its relative-path
checkpoint loading):
    python run_gat_inference.py [object_name ...]   # default: all 10 known objects
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(HERE)
AFF_GRASP_ROOT = os.path.join(REPO_ROOT, "Aff-Grasp", "affordance-learning")
DATA_DIR = os.path.join(REPO_ROOT, "regions_test", "data")
FIGURES_DIR = os.path.join(HERE, "figures", "gat_comparison")
os.makedirs(FIGURES_DIR, exist_ok=True)

CROP_SIZE = 448
CHECKPOINT = os.environ.get(
    "GAT_CHECKPOINT",
    "/tmp/claude-1325177/-shared-nas-data-m1-adadt2-nvrd/1599d579-452f-440d-8298-20f3ff309a9e/scratchpad/"
    "aff_grasp_ckpt/best_8_iou_68.4.pth",
)
KNOWN_OBJECTS = ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
                 "fire-alarm", "wallet"]
SPLITS = ("known", "novel", "shape-shape", "shape-texture")
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127)]  # one per AFF_LIST class, tab10-ish


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    """Normalize a raw depth map to 0-255 and replicate to 3 channels (grayscale, plain RGB for the processor)."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    gray = (d * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


def resolve_image_and_depth_path(object_name: str) -> tuple:
    """Depth maps are cached per split, alongside this project's own <split>_depth folders; if a split's depth folder doesn't have this object's depth
    map cached yet, it's generated on the fly with the same Depth-Anything model GAT itself uses."""
    for split in SPLITS:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(DATA_DIR, split, f"{object_name}{ext}")
            if os.path.isfile(candidate):
                depth_dir = os.path.join(DATA_DIR, f"{split}_depth")
                return candidate, os.path.join(depth_dir, f"{object_name}.png"), depth_dir
    raise FileNotFoundError(object_name)


def ensure_depth_map(image_path: str, depth_path: str, depth_dir: str):
    if os.path.isfile(depth_path):
        return
    from transformers import pipeline
    os.makedirs(depth_dir, exist_ok=True)
    print(f"  generating missing depth map -> {depth_path}")
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                           device="cpu")
    image = Image.open(image_path).convert("RGB")
    result = depth_pipe(image)
    depth_to_rgb(np.array(result["depth"])).save(depth_path)


def load_model(device):
    sys.path.insert(0, AFF_GRASP_ROOT)
    os.chdir(AFF_GRASP_ROOT)  # GAT.py loads 'dinov2_vitb14_pretrain.pth' via a bare relative path
    from models.GAT import Net
    from data.ego_video_data import AFF_LIST

    model = Net().to(device)
    state = torch.load(CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"[warning] {len(missing)} missing keys not in checkpoint (likely untrained/discarded "
              f"depth_embedder sublayer per the paper): {missing}")
    if unexpected:
        print(f"[warning] {len(unexpected)} unexpected keys in checkpoint: {unexpected}")
    model.eval()
    return model, AFF_LIST


def preprocess(image: Image.Image, depth: Image.Image):
    img_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE), antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dep_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE), antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return img_tf(image).unsqueeze(0), dep_tf(depth).unsqueeze(0)


def colorize(pred: np.ndarray, aff_list: list, threshold: float = 0.5) -> np.ndarray:
    """pred: (num_aff, H, W) in [0,1] (clamped model output). argmax class per pixel, background
    (all classes below threshold) rendered black."""
    max_idx = pred.argmax(axis=0)
    max_val = pred.max(axis=0)
    out = np.zeros((*pred.shape[1:], 3), dtype=np.uint8)
    for i in range(len(aff_list)):
        out[max_idx == i] = PALETTE[i]
    out[max_val < threshold] = (0, 0, 0)
    return out


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else KNOWN_OBJECTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, aff_list = load_model(device)
    print(f"AFF_LIST (GAT's vocabulary): {aff_list}")

    for obj in objects:
        img_path, dep_path, depth_dir = resolve_image_and_depth_path(obj)
        ensure_depth_map(img_path, dep_path, depth_dir)
        image = Image.open(img_path).convert("RGB")
        depth = Image.open(dep_path).convert("RGB")
        img_t, dep_t = preprocess(image, depth)

        with torch.no_grad():
            pred = model(img_t.to(device), dep_t.to(device))  # (1, num_aff, 448, 448)
        pred = pred.squeeze(0).cpu().numpy()

        seg = colorize(pred, aff_list)
        seg_img = Image.fromarray(seg).resize(image.size, Image.NEAREST)

        # Raw per-pixel prediction at native resolution, for scripts that need to read off GAT's
        # class for an arbitrary region mask (e.g. the Prolific affordance-study prep) rather than
        # just the colorized figure.
        pred_native = np.stack([
            np.array(Image.fromarray(ch).resize(image.size, Image.BILINEAR)) for ch in pred
        ])
        np.savez(os.path.join(FIGURES_DIR, f"{obj}_gat_raw.npz"), pred=pred_native, aff_list=aff_list)

        fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[1].imshow(seg_img)
        axs[1].set_title("GAT prediction (argmax per pixel)")
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in PALETTE[:len(aff_list)]]
        fig.legend(handles, aff_list, loc="lower center", ncol=len(aff_list), fontsize=9,
                   bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(f"{obj}  |  GAT (Aff-Grasp) affordance prediction", fontsize=13)
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"{obj}_gat_prediction.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"{obj}: saved {out_path}  (max class scores per channel: "
              f"{dict(zip(aff_list, pred.reshape(len(aff_list), -1).max(axis=1).round(2)))})")


if __name__ == "__main__":
    main()
