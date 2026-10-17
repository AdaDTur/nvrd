"""
Human-scale size-comparison charts for NVRD novel objects, in the style of the familiar
"dinosaur vs. human" silhouette charts: a solid-color object silhouette next to a fixed-height
human figure, on a labeled real-world scale grid -- so part-level affordance judgments (which
depend heavily on how big the object actually is) have a size reference to anchor against.

The per-object silhouette is deliberately NOT generative -- no image model is asked to draw or
imagine any of the 10 novel objects. Each one is extracted from its own existing render via
classical computer vision (cv2.grabCut: a 2004-era graph-cut foreground/background segmentation
algorithm, not a learned/neural model -- chosen over simple color-distance thresholding or
flood-fill because the renders have a soft studio lighting gradient across the backdrop, which a
single background-color seed doesn't capture well -- verified directly: flood-fill from the
corners leaked through the gradient and mis-segmented ~99% of the frame, grabCut didn't).

The human reference figure is the one exception: a single flat pictogram-style silhouette,
generated ONCE via Gemini 2.5 Flash Image (assets/human_silhouette_raw.png -> thresholded and
cached as assets/human_silhouette.png, same extraction idea as the objects but simple black/white
thresholding since the generation prompt controls contrast directly) and reused unchanged across
every chart for every object. This is a fundamentally different and much lower-risk use of a
generative model than editing a specific object: "draw a generic standing-person icon" is a task
image models are reliably good at, unlike precisely reshaping one particular novel object while
preserving its identity. If assets/human_silhouette.png is ever deleted or needs a different
pose/style, regenerate it with generate_human_asset.py.

For each object, three charts are generated at three hypothetical real-world sizes -- the
object's longest silhouette dimension is rescaled to:
    hand-size:     6 in    (0.15 m)
    human-size:    4.5 ft  (1.37 m -- chair/small-table scale, i.e. an object sized for a person
                            to interact with at arm's reach, NOT an object as tall as a person)
    building-size: 10 ft   (3.048 m)
The human figure itself is always drawn at a fixed real height (5'9" / 1.75 m) for scale
reference; only the grid's real-world-per-pixel scale and the object's rendered size change
between the three charts for a given object, matching how the reference dinosaur/raptor charts
keep their scale bar constant while the compared creature's size is what actually varies. Each
figure's actual height is called out with its own dashed arrow + feet'inches" label (both figures
use the same x'y" format -- a stray decimal-feet label on just one of the two was confusing).

Usage:
    python3 generate_size_charts.py [object ...]   # default: shape_1 only, for review
Output: figures/size_charts/<object>/<object>_<scenario>.png (hand, human, building)
"""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
_VENV_DIR = os.path.join(HERE, "venv_vlm")  # cv2/numpy/pillow already live here (labeling pipeline)
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(_venv_py, [_venv_py] + sys.argv)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(HERE)
NOVEL_DIR = os.path.join(REPO_ROOT, "regions_test", "data", "novel")
OUT_DIR = os.path.join(HERE, "figures", "size_charts")
os.makedirs(OUT_DIR, exist_ok=True)

HUMAN_HEIGHT_M = 1.75  # 5'9" -- the one constant across all three charts for a given object
M_PER_FT = 0.3048

# (label, target longest-dimension in meters) -- grid step is computed dynamically per chart
# (nice_step()) rather than fixed per scenario, since a fixed meters-per-gridline step that looks
# right at building scale produces dozens of illegibly packed gridlines at hand scale.
SCENARIOS = [
    ("hand", 0.15),      # 6 in object
    ("human", 1.37),     # ~4.5 ft object -- chair/small-table scale
    ("building", 3.048),  # 10 ft object
]

# Every chart is rendered at this fixed HEIGHT, matching the fixed square aspect-ratio the study
# page's other two image boxes (All Parts / Highlighted Part) use -- the display frame is a fixed
# square with object-fit: contain, so a chart this height always sits at the same displayed height
# as the other two, regardless of its width. WIDTH is computed dynamically per chart in
# render_chart() instead of being fixed square -- a wide object (or a long human-vs-object gap at
# some scales) can need more horizontal room than a 1:1 canvas has, and forcing it into a fixed
# square was clipping the content at the right edge (confirmed on shape_10 at building scale, the
# object's silhouette ran past the canvas boundary). A wider-than-square chart just gets a little
# letterbox padding top/bottom in the display frame instead -- never cropped.
CHART_HEIGHT = 1000

BG_COLOR = (250, 249, 246)
GRID_COLOR = (210, 206, 196)
BASELINE_COLOR = (120, 115, 105)
HUMAN_COLOR = (130, 170, 150)
OBJECT_COLOR = (143, 61, 26)  # matches this project's established "affordance-word" accent color
TEXT_COLOR = (60, 58, 54)


def extract_silhouette(image_path: str) -> np.ndarray:
    """Returns a cropped boolean mask (True = object) via cv2.grabCut, cleaned up with a small
    morphological close (fills anti-aliasing/specular-highlight gaps) and largest-connected-
    component selection (drops stray specks grabCut sometimes keeps at the frame edges).

    Seeded with a 3-zone mask (GC_INIT_WITH_MASK), not the simpler rect-init (GC_INIT_WITH_RECT)
    used originally -- rect-init marks the ENTIRE interior as "probable foreground" from the start,
    which includes the soft cast shadow these renders have at the object's base (same hue as the
    backdrop, just darker -- confirmed by direct pixel sampling, a shadow and a same-toned object
    surface aren't reliably separable by color alone). That foreground-leaning prior was enough to
    pull the shadow in as a thin "skirt" on several objects (visually confirmed: shape_1, shape_10,
    shape_14, shape_16 all had one). Seeding only a tight central core as probable-foreground, with
    a probable-BACKGROUND prior everywhere else (including right up against the object's true
    edges), removes that bias -- grabCut's own color-distribution modeling still correctly grows the
    foreground back out to the object's real extent (verified: thin spiky legs on shape_2/16/27 stay
    fully intact), it just no longer defaults to "probably object" in the ambiguous shadow zone."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    CORE_FRAC = 0.30  # half-width/height of the seeded "probable foreground" core, as a fraction
    # of the full frame -- these renders center the object consistently enough that a fixed
    # fraction works across all of them, without needing per-object tuning.
    OUTER_MARGIN_FRAC = 0.06  # thin border forced to "definite background"
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    core_h, core_w = int(h * CORE_FRAC), int(w * CORE_FRAC)
    mask[cy - core_h : cy + core_h, cx - core_w : cx + core_w] = cv2.GC_PR_FGD
    outer_margin = int(min(h, w) * OUTER_MARGIN_FRAC)
    mask[:outer_margin, :] = cv2.GC_BGD
    mask[-outer_margin:, :] = cv2.GC_BGD
    mask[:, :outer_margin] = cv2.GC_BGD
    mask[:, -outer_margin:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
    foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        foreground = (labels == largest).astype(np.uint8)

    ys, xs = np.nonzero(foreground)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return foreground[y0:y1, x0:x1].astype(bool)


def silhouette_to_rgba(mask: np.ndarray, color: tuple) -> Image.Image:
    h, w = mask.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask] = (*color, 255)
    return Image.fromarray(arr, mode="RGBA")


HUMAN_ASSET_PATH = os.path.join(HERE, "assets", "human_silhouette.png")
_human_mask_cache = None


def _human_mask() -> np.ndarray:
    """Cached boolean mask of the generated human silhouette asset (see generate_human_asset.py),
    cropped to its bounding box -- loaded once per process, reused (just resized+recolored) for
    every figure across every chart."""
    global _human_mask_cache
    if _human_mask_cache is None:
        if not os.path.isfile(HUMAN_ASSET_PATH):
            raise FileNotFoundError(
                f"{HUMAN_ASSET_PATH} not found -- run generate_human_asset.py first."
            )
        rgba = np.array(Image.open(HUMAN_ASSET_PATH).convert("RGBA"))
        _human_mask_cache = rgba[:, :, 3] > 127
    return _human_mask_cache


def draw_human(height_px: int, color: tuple) -> Image.Image:
    """The cached human silhouette asset, recolored and resized to the requested pixel height."""
    mask = _human_mask()
    rgba = silhouette_to_rgba(mask, color)
    mh, mw = mask.shape
    width_px = max(1, round(height_px * mw / mh))
    return rgba.resize((width_px, height_px), Image.LANCZOS)


def nice_step(span_m: float, target_lines: int = 6) -> float:
    """A 'round' grid step (1/2/5 x a power of ten, in meters) that gives roughly target_lines
    gridlines across span_m -- the standard chart-axis trick, needed because a step size tuned to
    look right at building scale produces dozens of illegibly packed gridlines at hand scale."""
    raw = span_m / target_lines
    magnitude = 10 ** np.floor(np.log10(raw))
    for mult in (1, 2, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude  # unreachable, mult=10 above always satisfies raw <= 10*magnitude


def format_feet_inches(m: float) -> str:
    """x'y" for both figures -- a human labeled "5'9"" next to an object labeled in decimal feet
    read as two different units at a glance, even though both were technically feet."""
    total_in = round(m / M_PER_FT * 12)
    ft, inch = divmod(total_in, 12)
    if ft == 0:
        return f'{inch}"'
    return f"{ft}'{inch}\""


def draw_height_arrow(draw: ImageDraw.ImageDraw, x: float, y_top: float, y_bottom: float,
                       label: str, font: ImageFont.ImageFont) -> None:
    """A dashed vertical double-ended arrow with a horizontal tick at each end and the size label
    at its vertical midpoint -- the same "caliper" convention the reference dino/raptor charts use
    for their height/length call-outs, so each figure gets its own explicit size label rather than
    relying solely on the shared background grid."""
    dash, gap = 7, 5
    y = y_top
    while y < y_bottom:
        y2 = min(y + dash, y_bottom)
        draw.line([(x, y), (x, y2)], fill=TEXT_COLOR, width=2)
        y += dash + gap
    tick = 7
    draw.line([(x - tick, y_top), (x + tick, y_top)], fill=TEXT_COLOR, width=2)
    draw.line([(x - tick, y_bottom), (x + tick, y_bottom)], fill=TEXT_COLOR, width=2)
    draw.text((x - tick - 6, (y_top + y_bottom) / 2), label, fill=TEXT_COLOR, font=font, anchor="rm")


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_chart(object_name: str, mask: np.ndarray, scenario_label: str, target_m: float,
                  height: int = CHART_HEIGHT) -> Image.Image:
    H = height

    # No title row anymore -- the study page's own "Size Reference" image-box label already says
    # what this is; a second title baked into the pixels was redundant.
    margin_top, margin_bottom, margin_side = 40, 90, 50
    plot_h = H - margin_top - margin_bottom
    baseline_y = margin_top + plot_h

    mask_h, mask_w = mask.shape
    # target_m is the LONGEST silhouette dimension; object_height_m is its vertical extent, which
    # equals target_m only when the object is taller than it is wide.
    object_height_m = target_m if mask_h >= mask_w else target_m * (mask_h / mask_w)
    # px-per-meter chosen so the taller of {human, object} fills ~80% of the plot height -- this
    # only depends on H (fixed), not on the canvas width computed below.
    px_per_m = (plot_h * 0.8) / max(HUMAN_HEIGHT_M, object_height_m)

    # Each figure gets LABEL_RESERVE px of room for its arrow's size label, then ARROW_GAP px
    # before the figure itself -- so neither the dashed arrow nor its text collides with the
    # silhouette standing next to it.
    LABEL_RESERVE, ARROW_GAP, FIGURE_GAP = 60, 18, 40
    name_font, label_font = _font(15), _font(15)

    human_h_px = HUMAN_HEIGHT_M * px_per_m
    human_img = draw_human(int(human_h_px), HUMAN_COLOR)

    object_h_px = object_height_m * px_per_m
    object_w_px = object_h_px * (mask_w / mask_h)
    object_img = Image.fromarray(
        np.array(silhouette_to_rgba(mask, OBJECT_COLOR))
    ).resize((max(1, int(object_w_px)), max(1, int(object_h_px))), Image.LANCZOS)

    # The whole content block (human's label+arrow+figure, a gap, the object's label+arrow+figure)
    # is measured as one unit first, and the canvas is sized to fit it -- at least as wide as it is
    # tall (so compact charts stay the same square the display frame expects), but wider when the
    # content itself needs more room, so nothing ever runs past the canvas edge the way a fixed
    # square canvas could (confirmed clipped on shape_10 at building scale before this fix).
    human_block_w = LABEL_RESERVE + ARROW_GAP + human_img.width
    object_block_w = LABEL_RESERVE + ARROW_GAP + object_img.width
    total_content_w = human_block_w + FIGURE_GAP + object_block_w
    W = max(H, int(total_content_w + 2 * margin_side))

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # grid: unlabeled vertical lines at a "nice" interval (~6 lines) -- pure background texture,
    # like the reference charts' grid; the actual measurements live on each figure's own arrow,
    # not on the axis, so no per-line numbers here.
    usable_width_m = (W - 2 * margin_side) / px_per_m
    grid_step_m = nice_step(usable_width_m)
    grid_step_px = grid_step_m * px_per_m
    x = margin_side
    while x < W - margin_side + 1:
        draw.line([(x, margin_top), (x, baseline_y)], fill=GRID_COLOR, width=1)
        x += grid_step_px
    draw.line([(margin_side, baseline_y), (W - margin_side, baseline_y)], fill=BASELINE_COLOR, width=2)

    # content block centered in the (now content-fitted) canvas
    content_start_x = max(margin_side, (W - total_content_w) / 2)

    # human figure, with its own height arrow+label
    human_arrow_x = content_start_x + LABEL_RESERVE
    human_x = human_arrow_x + ARROW_GAP
    human_top_y = baseline_y - human_h_px
    canvas.paste(human_img, (int(human_x), int(human_top_y)), human_img)
    draw_height_arrow(draw, human_arrow_x, human_top_y, baseline_y, format_feet_inches(HUMAN_HEIGHT_M), label_font)
    draw.text(
        (human_x + human_img.width / 2, baseline_y + 30),
        "Person", fill=TEXT_COLOR, font=name_font, anchor="ma",
    )

    # object silhouette, scaled to this scenario's target size, right of the human, with its own
    # height arrow+label
    object_arrow_x = human_x + human_img.width + FIGURE_GAP + LABEL_RESERVE
    object_x = object_arrow_x + ARROW_GAP
    object_top_y = baseline_y - object_h_px
    canvas.paste(object_img, (int(object_x), int(object_top_y)), object_img)
    draw_height_arrow(draw, object_arrow_x, object_top_y, baseline_y, format_feet_inches(object_height_m), label_font)
    draw.text(
        (object_x + object_img.width / 2, baseline_y + 30),
        object_name, fill=TEXT_COLOR, font=name_font, anchor="ma",
    )

    return canvas


def main():
    objects = sys.argv[1:] if len(sys.argv) > 1 else ["shape_1"]
    for obj in objects:
        src = os.path.join(NOVEL_DIR, f"{obj}.png")
        if not os.path.isfile(src):
            print(f"  {obj}: no render at {src}")
            continue
        print(f"=== {obj} ===")
        mask = extract_silhouette(src)
        print(f"  silhouette: {mask.shape[1]}x{mask.shape[0]} px, {mask.mean():.1%} fill")

        obj_out_dir = os.path.join(OUT_DIR, obj)
        os.makedirs(obj_out_dir, exist_ok=True)
        for label, target_m in SCENARIOS:
            chart = render_chart(obj, mask, label, target_m)
            out_path = os.path.join(obj_out_dir, f"{obj}_{label}.png")
            chart.save(out_path)
            print(f"  {label}: saved {out_path} ({chart.width}x{chart.height})")


if __name__ == "__main__":
    main()
