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

        band1 = ds.GetRasterBand(1)
        if band1 is None:
            raise ValueError("Dataset has no raster bands")

        # Case A: single-band paletted/indexed image
        if ds.RasterCount == 1:
            color_table = band1.GetColorTable()
            arr = band1.ReadAsArray()
            if arr is None:
                raise ValueError("ReadAsArray returned None for band 1")

            if color_table is not None:
                count = color_table.GetCount()
                lut = np.zeros((count, 3), dtype=np.uint8)

                for i in range(count):
                    entry = color_table.GetColorEntry(i)
                    if entry is None:
                        continue
                    r, g, b, _a = entry
                    lut[i] = [r, g, b]

                idx = arr.astype(np.int32, copy=False)
                idx = np.clip(idx, 0, count - 1)
                return lut[idx]

            # True grayscale single-band
            if arr.dtype != np.uint8:
                arr = normalize_to_uint8(arr)
            return np.stack([arr, arr, arr], axis=-1)

        # Case B: multi-band image
        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("ReadAsArray returned None")

        if arr.ndim != 3:
            raise ValueError(f"Unsupported GDAL array shape: {arr.shape}")

        bands, _, _ = arr.shape
        if bands < 3:
            gray = arr[0]
            if gray.dtype != np.uint8:
                gray = normalize_to_uint8(gray)
            return np.stack([gray, gray, gray], axis=-1)

        rgb = np.transpose(arr[:3], (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = normalize_to_uint8(rgb)
        return rgb

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

    # Extract just (x, y) - handle both 2-tuple and 3-tuple points
    pts = np.asarray([(p[0], p[1]) for p in points], dtype=np.float32)

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



def compute_strip_weights(points, orientation, total_strips):
    """
    Compute weights for points based on strip distance from edges.
    Strips near edges (where frame is strongest) get higher weight.
    
    Points should be tuples: (x, y, strip_index)
    Returns: numpy array of weights [0.5 .. 1.0]
    """
    if not points or len(points[0]) < 3:
        return np.ones(len(points))
    
    strip_indices = np.array([p[2] for p in points])
    center = total_strips / 2.0
    
    # Weight formula: 1.0 at edges, 0.5 at center
    # distance = |i - center| / center
    # weight = 1.0 - 0.5 * distance
    distances = np.abs(strip_indices - center) / center


    # weights = 1.0 - 0.5 * np.clip(distances, 0, 1.0)

    # Make edge strips REALLY dominant (0.3 at center):
    # weights = 1.0 - 0.7 * np.clip(distances, 0, 1.0)

    # Softer weighting (0.8 at center):
    #weights = 1.0 - 0.2 * np.clip(distances, 0, 1.0)

    # Square-law falloff (more aggressive):
    #weights = (1.0 - np.clip(distances, 0, 1.0) ** 2)

    # make edges stronger than center
    weights = 0.3 + 0.7 * np.clip(distances, 0, 1.0)
    
    return weights

def fit_line_weighted(points, orientation, total_strips):
    """Fit y = m*x + c with point weights based on strip position."""
    if len(points) < 3: 
        return None
    
    # Extract (x, y) only, ignore strip index
    pts = np.array([(p[0], p[1]) for p in points])
    weights = compute_strip_weights(points, orientation, total_strips)
    
    if orientation == 'h': 
        X = pts[:, 0]
        y = pts[:, 1]
    else: 
        X = pts[:, 1]
        y = pts[:, 0]
    
    # Outlier removal (same as simple, on weighted data)
    median_val = np.median(y)
    mask = np.abs(y - median_val) < 50 
    clean_pts = pts[mask]
    clean_weights = weights[mask]
    
    if len(clean_pts) < 2: 
        return None
    
    # Weighted polyfit (numpy supports w parameter)
    if orientation == 'h':
        m, c = np.polyfit(clean_pts[:, 0], clean_pts[:, 1], 1, w=clean_weights)
    else:
        m, c = np.polyfit(clean_pts[:, 1], clean_pts[:, 0], 1, w=clean_weights)
    
    return m, c

def average_world_dimensions(world_coords):
    pts = np.asarray(world_coords, dtype=np.float32)

    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bot = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])

    return (w_top + w_bot) / 2.0, (h_left + h_right) / 2.0


def line_value(line, t):
    """Evaluate a fitted line at x=t for horizontal lines or y=t for vertical lines."""
    m, c = line
    return float(m * t + c)