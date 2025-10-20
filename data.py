
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List, Dict

RNG = np.random.default_rng(42)

def _class_palette(n_classes: int) -> np.ndarray:
    """
    Deterministic color palette (RGB) for up to 50 classes.
    """
    RNG_local = np.random.default_rng(1234)
    palette = (RNG_local.random((max(n_classes, 50), 3)) * 200 + 30).astype(np.uint8)
    return palette[:n_classes]

def _draw_pattern(img: Image.Image, cls_id: int, jitter: float = 0.15) -> None:
    """
    Draw a simple, class-specific geometric pattern so classes are learnable
    even in a tiny synthetic dataset.
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 3

    # Pattern family by cls_id mod 5
    mode = cls_id % 5
    if mode == 0:
        # Circle
        dx = int(r * jitter * (RNG.random() - 0.5) * 2)
        dy = int(r * jitter * (RNG.random() - 0.5) * 2)
        draw.ellipse([cx - r + dx, cy - r + dy, cx + r + dx, cy + r + dy], outline=255, width=2)
    elif mode == 1:
        # Rectangle
        dx = int(r * 0.6); dy = int(r * 0.4)
        draw.rectangle([cx - dx, cy - dy, cx + dx, cy + dy], outline=255, width=2)
    elif mode == 2:
        # Triangle
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(pts, outline=255)
    elif mode == 3:
        # Cross lines
        draw.line([0, cy, w, cy], fill=255, width=2)
        draw.line([cx, 0, cx, h], fill=255, width=2)
    else:
        # Diagonal stripes
        step = max(3, w // 6)
        for x in range(-w, 2 * w, step):
            draw.line([x, 0, x + w, h], fill=255, width=1)

def _render_sample(image_size: int, color: Tuple[int, int, int], cls_id: int) -> np.ndarray:
    bg = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    # base colored blob
    blob = Image.new("RGBA", (image_size, image_size), (*color, 0))
    draw = ImageDraw.Draw(blob)
    # Irregular blob via jittered ellipse
    rx = int(image_size * (0.35 + 0.1 * (RNG.random() - 0.5)))
    ry = int(image_size * (0.28 + 0.1 * (RNG.random() - 0.5)))
    cx = int(image_size * (0.5 + 0.1 * (RNG.random() - 0.5)))
    cy = int(image_size * (0.5 + 0.1 * (RNG.random() - 0.5)))
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(*color, 220))
    blob = blob.filter(ImageFilter.GaussianBlur(0.8 + RNG.random()*0.8))
    bg.paste(blob, (0, 0), blob)

    # Add class-specific white overlay pattern for separability
    _draw_pattern(bg, cls_id)

    # mild jitter: brightness/contrast
    arr = np.asarray(bg).astype(np.float32)
    brightness = 0.9 + 0.2 * (RNG.random() - 0.5)
    contrast = 0.9 + 0.2 * (RNG.random() - 0.5)
    arr = (arr - 127.5) * contrast + 127.5
    arr = arr * brightness
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def generate_dataset(
    n_classes: int = 20,
    samples_per_class: int = 15,
    image_size: int = 28,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns (X, y, class_names)
    X: [N, H, W, 3] uint8
    y: [N] int labels in [0, n_classes)
    """
    palette = _class_palette(n_classes)
    class_names = [
        "Subol_Lota", "Bashmoti", "Ganjiya", "Shampakatari", "Katarivog",
        "BR28", "BR29", "Paijam", "Bashful", "Lal_Aush",
        "Jirashail", "Gutisharna", "Red_Cargo", "Najirshail", "Katari_Polao",
        "Lal_Biroi", "Chinigura_Polao", "Amon", "Shorna5", "Lal_Binni"
    ][:n_classes]

    X_list = []
    y_list = []

    for cls in range(n_classes):
        color = tuple(int(c) for c in palette[cls])
        for _ in range(samples_per_class):
            img = _render_sample(image_size, color, cls)
            X_list.append(img)
            y_list.append(cls)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, class_names
