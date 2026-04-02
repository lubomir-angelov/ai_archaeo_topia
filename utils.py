import logging
from pathlib import Path
import numpy as np
import cv2
from osgeo import gdal

logger = logging.getLogger(__name__)


def read_image_gray_any(image_path: str | Path) -> np.ndarray:
    """Read an image as grayscale using OpenCV first, then GDAL fallback.

    Returns:
        2D uint8 grayscale image.

    Raises:
        ValueError: If the image cannot be read by any backend.
    """
    image_path = str(image_path)

    # 1) Standard OpenCV path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img
    except Exception as exc:
        logger.warning("OpenCV grayscale read failed for %s: %s", image_path, exc)

    # 2) OpenCV + GDAL path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img
    except Exception as exc:
        logger.warning("OpenCV GDAL grayscale read failed for %s: %s", image_path, exc)

    # 3) Native GDAL path
    try:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError("gdal.Open returned None")

        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("ReadAsArray returned None")

        # GDAL returns:
        # - 2D array for single-band
        # - 3D array shaped (bands, rows, cols) for multi-band
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3:
            if arr.shape[0] >= 3:
                rgb = np.transpose(arr[:3], (1, 2, 0))
                if rgb.dtype != np.uint8:
                    rgb = normalize_to_uint8(rgb)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr[0]
        else:
            raise ValueError(f"Unsupported GDAL array shape: {arr.shape}")

        if gray.dtype != np.uint8:
            gray = normalize_to_uint8(gray)

        return gray

    except Exception as exc:
        raise ValueError(f"Could not read image with OpenCV or GDAL: {exc}") from exc


def read_image_color_any(image_path: str | Path) -> np.ndarray:
    """Read an image as RGB using OpenCV first, then GDAL fallback.

    Returns:
        3D uint8 RGB image.

    Raises:
        ValueError: If the image cannot be read by any backend.
    """
    image_path = str(image_path)

    # 1) Standard OpenCV path
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("OpenCV color read failed for %s: %s", image_path, exc)

    # 2) OpenCV + GDAL path
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("OpenCV GDAL color read failed for %s: %s", image_path, exc)

    # 3) Native GDAL path
    try:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError("gdal.Open returned None")

        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("ReadAsArray returned None")

        if arr.ndim == 2:
            gray = normalize_to_uint8(arr) if arr.dtype != np.uint8 else arr
            return np.stack([gray, gray, gray], axis=-1)

        if arr.ndim == 3:
            bands, _, _ = arr.shape
            if bands >= 3:
                rgb = np.transpose(arr[:3], (1, 2, 0))
                if rgb.dtype != np.uint8:
                    rgb = normalize_to_uint8(rgb)
                return rgb
            gray = arr[0]
            if gray.dtype != np.uint8:
                gray = normalize_to_uint8(gray)
            return np.stack([gray, gray, gray], axis=-1)

        raise ValueError(f"Unsupported GDAL array shape: {arr.shape}")

    except Exception as exc:
        raise ValueError(f"Could not read image with OpenCV or GDAL: {exc}") from exc


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize numeric raster data to uint8 for CV processing."""
    arr = np.asarray(arr)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    min_val = float(arr[finite_mask].min())
    max_val = float(arr[finite_mask].max())

    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr.astype(np.float32) - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def robust_fit_line(points, orientation, residual_thresh, iterations=200):
    """Fit y = m*x + c for horizontal-like lines, x = m*y + c for vertical-like lines."""
    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float32)

    if orientation == 'h':
        indep = pts[:, 0]  # x
        dep = pts[:, 1]    # y
    else:
        indep = pts[:, 1]  # y
        dep = pts[:, 0]    # x

    n = len(indep)
    rng = np.random.default_rng(42)
    best_mask = None
    best_count = 0

    if n == 2:
        best_mask = np.array([True, True])
    else:
        for _ in range(iterations):
            i, j = rng.choice(n, size=2, replace=False)
            x1, x2 = indep[i], indep[j]
            if abs(x2 - x1) < 1e-6:
                continue

            m = (dep[j] - dep[i]) / (x2 - x1)
            c = dep[i] - m * x1

            residuals = np.abs(dep - (m * indep + c))
            mask = residuals < residual_thresh
            count = int(mask.sum())

            if count > best_count:
                best_count = count
                best_mask = mask

    if best_mask is None or best_mask.sum() < 2:
        return None

    m, c = np.polyfit(indep[best_mask], dep[best_mask], 1)
    return float(m), float(c)



def rectangle_sanity_score(pixel_coords, image_w, image_h):
    pts = np.array(pixel_coords, dtype=np.float32)

    # Side lengths
    top = np.linalg.norm(pts[1] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])

    # Penalize out-of-bounds corners
    oob = 0.0
    for x, y in pts:
        if x < 0 or x >= image_w or y < 0 or y >= image_h:
            oob += 1000.0

    # Penalize asymmetry
    width_mismatch = abs(top - bottom) / max(top, bottom, 1.0)
    height_mismatch = abs(left - right) / max(left, right, 1.0)

    # Penalize bizarre slopes / twisted geometry
    diag1 = np.linalg.norm(pts[2] - pts[0])
    diag2 = np.linalg.norm(pts[3] - pts[1])
    diag_mismatch = abs(diag1 - diag2) / max(diag1, diag2, 1.0)

    return oob + 100.0 * width_mismatch + 100.0 * height_mismatch + 100.0 * diag_mismatch


def filter_points_by_position(points, orientation, image_w, image_h):
    filtered = []

    for x, y in points:
        if orientation == 'h':
            # For top and bottom, y matters
            if 0 <= y < image_h:
                filtered.append((x, y))
        else:
            # For left and right, x matters
            if 0 <= x < image_w:
                filtered.append((x, y))

    return filtered


def preprocess_for_line_detection(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)

    bw = cv2.adaptiveThreshold(
        eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    return bw

def mean_corner_distance(a, b):
    pa = np.array(a, dtype=np.float32)
    pb = np.array(b, dtype=np.float32)
    
    return float(np.mean(np.linalg.norm(pa - pb, axis=1)))


def build_side_variants(point_sets_by_rank, orientation, residual_thresh, side_name):
    variants = []

    for rank, pts in enumerate(point_sets_by_rank):
        if len(pts) < 3:
            continue

        line_simple = fit_line_simple(pts, orientation)
        if line_simple is not None:
            variants.append({
                "name": f"{side_name}_r{rank}_simple",
                "side": side_name,
                "rank": rank,
                "method": "simple",
                "line": line_simple,
                "points": pts,
            })

        line_rob = robust_fit_line(pts, orientation, residual_thresh)
        if line_rob is not None:
            variants.append({
                "name": f"{side_name}_r{rank}_robust",
                "side": side_name,
                "rank": rank,
                "method": "robust",
                "line": line_rob,
                "points": pts,
            })

    return variants


def fit_line_simple(points, orientation):
    if len(points) < 3: 
        return None
    pts = np.array(points)
    
    if orientation == 'h': 
        X = pts[:, 0]
        y = pts[:, 1]
    else: 
        X = pts[:, 1]
        y = pts[:, 0]
        
    median_val = np.median(y)
    mask = np.abs(y - median_val) < 50 
    clean_pts = pts[mask]
    
    if len(clean_pts) < 2: 
        return None
    
    if orientation == 'h':
        m, c = np.polyfit(clean_pts[:, 0], clean_pts[:, 1], 1)
    else:
        m, c = np.polyfit(clean_pts[:, 1], clean_pts[:, 0], 1) 
        
    return m, c