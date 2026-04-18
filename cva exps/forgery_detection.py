import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


# -----------------------------
# Image Loading and Conversion
# -----------------------------
def load_image(image_path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert PIL image to a grayscale NumPy array (uint8)."""
    gray = image.convert("L")
    return np.array(gray, dtype=np.uint8)


# -----------------------------
# Point Transformations
# -----------------------------
def negative_transform(gray: np.ndarray) -> np.ndarray:
    """Apply negative transformation: s = 255 - r."""
    return 255 - gray


def contrast_adjustment(gray: np.ndarray, in_low=30, in_high=220) -> np.ndarray:
    """Simple linear contrast stretch.

    Values below in_low are clipped to 0.
    Values above in_high are clipped to 255.
    """
    gray_f = gray.astype(np.float32)
    stretched = (gray_f - in_low) * (255.0 / max(in_high - in_low, 1))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)


def threshold_transform(gray: np.ndarray, threshold=130) -> np.ndarray:
    """Binary thresholding for emphasizing abrupt intensity transitions."""
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def enhance_image(gray: np.ndarray) -> np.ndarray:
    """Combine point transformations to improve visibility.

    Strategy:
    1) Contrast adjustment to expand visible range.
    2) Mild blend with negative image to expose hidden edits.
    3) Light threshold-guided boost for abrupt transitions.
    """
    contrast = contrast_adjustment(gray)
    negative = negative_transform(gray)
    binary = threshold_transform(contrast, threshold=140)

    # Blend for visibility; values chosen for beginner-friendly behavior.
    enhanced = (0.65 * contrast + 0.25 * negative + 0.10 * binary).astype(np.float32)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# -----------------------------
# Histogram Equalization
# -----------------------------
def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Perform histogram equalization manually using CDF."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)

    # Normalize CDF to [0, 255]
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min() + 1e-9)
    cdf_final = np.ma.filled(cdf_norm, 0).astype(np.uint8)

    return cdf_final[gray]


# -----------------------------
# Edge Detection (Sobel, Prewitt)
# -----------------------------
def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution with edge padding (for educational use)."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def gradient_magnitude(gray: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Compute edge magnitude from x/y gradient kernels."""
    gray_f = gray.astype(np.float32)
    gx = convolve2d(gray_f, kx)
    gy = convolve2d(gray_f, ky)
    mag = np.sqrt(gx**2 + gy**2)

    # Normalize to 0-255 for visualization and thresholding.
    mag = 255 * (mag / (mag.max() + 1e-9))
    return mag.astype(np.uint8)


def detect_edges(equalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate Sobel and Prewitt edge maps."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    sobel = gradient_magnitude(equalized, sobel_x, sobel_y)
    prewitt = gradient_magnitude(equalized, prewitt_x, prewitt_y)
    return sobel, prewitt


# -----------------------------
# Forgery Analysis
# -----------------------------
def intensity_transition_map(equalized: np.ndarray) -> np.ndarray:
    """Estimate abrupt intensity transitions using finite differences."""
    eq = equalized.astype(np.float32)

    dx = np.zeros_like(eq)
    dy = np.zeros_like(eq)
    dx[:, 1:] = np.abs(eq[:, 1:] - eq[:, :-1])
    dy[1:, :] = np.abs(eq[1:, :] - eq[:-1, :])

    grad = np.sqrt(dx**2 + dy**2)
    grad = 255 * (grad / (grad.max() + 1e-9))
    return grad.astype(np.uint8)


def connected_components_boxes(mask: np.ndarray, min_area: int = 80) -> list[tuple[int, int, int, int]]:
    """Find connected components and return bounding boxes.

    Box format: (x_min, y_min, x_max, y_max)
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True

            pixels = []
            y_min = y_max = y
            x_min = x_max = x

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))

                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] != 0:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if len(pixels) >= min_area:
                boxes.append((x_min, y_min, x_max, y_max))

    return boxes


def overlay_and_boxes(
    original_rgb: np.ndarray,
    suspicious_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Create visual output with red overlay + yellow bounding boxes."""
    base = original_rgb.astype(np.float32).copy()

    # Red overlay on suspicious pixels.
    red_layer = np.zeros_like(base)
    red_layer[..., 0] = 255

    mask_3 = np.stack([suspicious_mask > 0] * 3, axis=-1)
    base[mask_3] = (1 - alpha) * base[mask_3] + alpha * red_layer[mask_3]
    overlay_img = np.clip(base, 0, 255).astype(np.uint8)

    # Draw bounding boxes around connected suspicious regions.
    pil_overlay = Image.fromarray(overlay_img)
    draw = ImageDraw.Draw(pil_overlay)
    for x_min, y_min, x_max, y_max in boxes:
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(255, 255, 0), width=2)

    return np.array(pil_overlay)


def detect_forgery(
    original_rgb: np.ndarray,
    equalized: np.ndarray,
    sobel: np.ndarray,
    prewitt: np.ndarray,
    edge_threshold: int = 120,
    irregularity_threshold: int = 45,
    transition_threshold: int = 110,
    min_area: int = 90,
    overlay_alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Build a suspicious mask and final highlighted visualization.

    Suspicious areas are where:
    - strong edges exist,
    - Sobel and Prewitt disagree noticeably (irregular edges), and/or
    - abrupt intensity transitions are high.
    """
    strong_edges = ((sobel > edge_threshold) | (prewitt > edge_threshold)).astype(np.uint8)
    irregular_edges = (np.abs(sobel.astype(np.int16) - prewitt.astype(np.int16)) > irregularity_threshold).astype(np.uint8)

    transitions = intensity_transition_map(equalized)
    unnatural_transitions = (transitions > transition_threshold).astype(np.uint8)

    suspicious = ((strong_edges & irregular_edges) | unnatural_transitions).astype(np.uint8)

    # Remove tiny isolated noise by keeping only components above area threshold.
    boxes = connected_components_boxes(suspicious, min_area=min_area)
    clean_mask = np.zeros_like(suspicious, dtype=np.uint8)
    for x_min, y_min, x_max, y_max in boxes:
        clean_mask[y_min : y_max + 1, x_min : x_max + 1] = np.maximum(
            clean_mask[y_min : y_max + 1, x_min : x_max + 1],
            suspicious[y_min : y_max + 1, x_min : x_max + 1],
        )

    final_img = overlay_and_boxes(original_rgb, clean_mask, boxes, alpha=overlay_alpha)
    return clean_mask * 255, final_img, boxes


# -----------------------------
# Visualization
# -----------------------------
def show_results(
    original_rgb: np.ndarray,
    enhanced: np.ndarray,
    equalized: np.ndarray,
    sobel: np.ndarray,
    prewitt: np.ndarray,
    final_highlighted: np.ndarray,
    suspicious_count: int,
) -> None:
    """Display outputs in required subplot format."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced, cmap="gray")
    axes[0, 1].set_title("Enhanced Image (Point Transforms)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(equalized, cmap="gray")
    axes[0, 2].set_title("Equalized Image")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(sobel, cmap="gray")
    axes[1, 0].set_title("Sobel Edge Map")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(prewitt, cmap="gray")
    axes[1, 1].set_title("Prewitt Edge Map")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(final_highlighted)
    axes[1, 2].set_title(f"Forgery Highlighted (Overlay + Boxes) | Regions: {suspicious_count}")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Demo + Main
# -----------------------------
def create_demo_image(size=(320, 240)) -> Image.Image:
    """Create a synthetic demo image with a tampered patch for quick testing."""
    w, h = size
    base = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient
    for y in range(h):
        val = int(60 + 140 * (y / max(h - 1, 1)))
        base[y, :, :] = [val, val, val]

    # Natural-looking object
    base[70:170, 50:140, :] = [120, 150, 130]

    # Simulated forged patch with abrupt boundaries
    base[95:165, 180:270, :] = [220, 220, 220]
    base[110:150, 200:250, :] = [30, 30, 30]

    return Image.fromarray(base)


def run_pipeline(
    image: Image.Image,
    edge_threshold: int,
    irregularity_threshold: int,
    transition_threshold: int,
    min_area: int,
    overlay_alpha: float,
) -> None:
    original_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = to_grayscale(image)

    enhanced = enhance_image(gray)
    equalized = histogram_equalization(enhanced)
    sobel, prewitt = detect_edges(equalized)
    _, final_highlighted, boxes = detect_forgery(
        original_rgb,
        equalized,
        sobel,
        prewitt,
        edge_threshold=edge_threshold,
        irregularity_threshold=irregularity_threshold,
        transition_threshold=transition_threshold,
        min_area=min_area,
        overlay_alpha=overlay_alpha,
    )

    print(f"Suspicious regions detected: {len(boxes)}")
    if len(boxes) == 0:
        print("No suspicious region detected with current thresholds.")
        print("Try lower values: --edge-threshold 70 --irregularity-threshold 20 --transition-threshold 70")

    show_results(original_rgb, enhanced, equalized, sobel, prewitt, final_highlighted, len(boxes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini Project: Digital Image Forgery Detection using image processing techniques"
    )
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with an internally generated demo image containing a forged region",
    )
    parser.add_argument("--edge-threshold", type=int, default=120, help="Strong edge threshold (default: 120)")
    parser.add_argument(
        "--irregularity-threshold",
        type=int,
        default=45,
        help="Sobel-Prewitt difference threshold (default: 45)",
    )
    parser.add_argument(
        "--transition-threshold",
        type=int,
        default=110,
        help="Intensity transition threshold (default: 110)",
    )
    parser.add_argument("--min-area", type=int, default=90, help="Minimum suspicious region area (default: 90)")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay visibility from 0.0 to 1.0 (default: 0.45)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        image = create_demo_image()
        run_pipeline(
            image,
            edge_threshold=args.edge_threshold,
            irregularity_threshold=args.irregularity_threshold,
            transition_threshold=args.transition_threshold,
            min_area=args.min_area,
            overlay_alpha=args.overlay_alpha,
        )
        return

    if not args.image:
        raise ValueError("Please provide --image <path> or run with --demo")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(str(image_path))
    run_pipeline(
        image,
        edge_threshold=args.edge_threshold,
        irregularity_threshold=args.irregularity_threshold,
        transition_threshold=args.transition_threshold,
        min_area=args.min_area,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
