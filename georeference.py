import argparse
import cv2
import numpy as np
from osgeo import gdal, osr
import os
import json
import re
import glob
import math
import csv
from pathlib import Path
from tqdm import tqdm  # Progress Bar

from utils import fit_line_weighted, mean_corner_distance, read_image_gray_any, read_image_color_any, robust_fit_line

# Enable GDAL Exceptions
gdal.UseExceptions()

# ==========================================
# CONFIGURATION
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent          

# PATHS
INPUT_FOLDER  = SCRIPT_DIR / "data/maps"
OUTPUT_FOLDER = SCRIPT_DIR / "data/georeferenced"
GEOJSON_PATH  = SCRIPT_DIR / "grid_25k.geojson"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
GEOJSON_NAME_FIELD = "mapsheet"
DEFAULT_EPSG = 25835

# ==========================================
# PART 0: DEBUGGING & VISUALIZATION
# ==========================================

def save_debug_overlay(
    image_path,
    pixel_coords,
    top_pts,
    bot_pts,
    left_pts,
    right_pts,
    out_path,
    debug_data,
):
    # Try color first
    try:
        img_rgb = read_image_color_any(image_path)   # your util returns RGB
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        gray = read_image_gray_any(image_path)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Top/bottom points = green
    for x, y in top_pts:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y in bot_pts:
        cv2.circle(img, (int(x), int(y)), 6, (0, 220, 0), -1)

    # Left/right points = blue
    for x, y in left_pts:
        cv2.circle(img, (int(x), int(y)), 6, (255, 0, 0), -1)
    for x, y in right_pts:
        cv2.circle(img, (int(x), int(y)), 6, (220, 0, 0), -1)

    # Final polygon = RED and thicker
    pts = np.array(pixel_coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=6)

    # Corner labels = yellow
    for i, (x, y) in enumerate(pixel_coords):
        label = f'{debug_data["best_candidate"]} | score={debug_data["best_score"]:.2f}'
        cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(
        #     img,
        #     f"C{i+1}",
        #     (int(x) + 10, int(y) - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )

    cv2.imwrite(out_path, img)

# ==========================================
# PART 1: DB & UTILS
# ==========================================

def transliterate_bg_to_en(text):
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f',
        'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sht', 'ъ': 'a',
        'ь': 'y', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ж': 'Zh',
        'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
        'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F',
        'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sht', 'Ъ': 'A',
        'Ь': 'Y', 'Ю': 'Yu', 'Я': 'Ya'
    }
    result = ""
    for char in text:
        result += translit_map.get(char, char)
    return result

def generate_match_keys(s):
    if not s: return []
    s_en = transliterate_bg_to_en(str(s))
    s_upper = s_en.upper()
    v1 = re.sub(r'[^A-Z0-9]', '', s_upper)
    parts = re.split(r'[^A-Z0-9]+', s_upper)
    norm = [str(int(p)) if p.isdigit() else p for p in parts if p]
    v2 = "".join(norm)
    return list(set([v1, v2]))

def get_closest_point(target_x, target_y, points):
    best_p = None
    min_dist = float('inf')
    for p in points:
        dist = math.hypot(p[0] - target_x, p[1] - target_y)
        if dist < min_dist:
            min_dist = dist
            best_p = p
    return best_p

def load_geojson_database(json_path, name_field):
    print(f"Loading DB: {json_path}")
    if not os.path.exists(json_path):
        print(f"ERROR: GeoJSON not found at {json_path}")
        return {}, DEFAULT_EPSG
        
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    epsg = DEFAULT_EPSG
    try:
        crs = data.get('crs', {}).get('properties', {}).get('name', '')
        m = re.search(r'EPSG[:]+(\d+)', crs, re.IGNORECASE)
        if m: 
            epsg = int(m.group(1))
    except: 
        pass

    db = {}
    for ft in data.get('features', []):
        props = ft.get('properties', {})
        name_bg = props.get(name_field, "")
        if not name_bg: continue
        
        name_en = transliterate_bg_to_en(name_bg)
        
        geom = ft.get('geometry', {})
        coords = []
        gtype = geom.get('type')
        raw = geom.get('coordinates', [])
        if gtype == 'Point': coords.append(raw)
        elif gtype == 'Polygon': coords.extend(raw[0])
        elif gtype == 'LineString': coords.extend(raw)
        elif gtype == 'MultiPolygon': 
            for p in raw: coords.extend(p[0])
        
        uniq = list(set([(float(p[0]), float(p[1])) for p in coords]))
        if len(uniq) < 4: continue
        
        xs = [p[0] for p in uniq]
        ys = [p[1] for p in uniq]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        tl = get_closest_point(min_x, max_y, uniq)
        tr = get_closest_point(max_x, max_y, uniq)
        br = get_closest_point(max_x, min_y, uniq)
        bl = get_closest_point(min_x, min_y, uniq)
        
        entry = {"original_name": name_en, "coords": [tl, tr, br, bl]}
        for k in generate_match_keys(name_en): db[k] = entry
    return db, epsg

# ==========================================
# PART 2: PROJECTION DETECTION LOGIC
# ==========================================

def find_line_in_strip_projection(strip, orientation, max_search_dist, mode='first_after_gap'):
    """
    Sums pixels along the non-scan axis. 
    Finds the frame line using projection profiles.
    
    mode='first_after_gap': Standard. Finds first strong line AFTER a large gap.
    """
    axis = 1 if orientation == 'h' else 0
    
    # Invert image (Black lines become bright peaks)
    prof = np.sum(255 - strip, axis=axis)
    max_val = np.max(prof)

    profile_len = len(prof)
    cluster_join_dist = max(2, int(profile_len * 0.01))
    large_gap_threshold = max(8, int(profile_len * 0.04))
    
    # Threshold to identify potential lines (35% of max intensity)
    threshold = max_val * 0.35 
    peaks = np.where(prof > threshold)[0]
    
    if len(peaks) == 0: 
        return None

    # Cluster peaks (group adjacent pixels into lines)
    clusters = []
    curr = [peaks[0]]
    for i in range(1, len(peaks)):
        #if peaks[i] <= peaks[i-1] + 5:
        if peaks[i] <= peaks[i - 1] + cluster_join_dist:
            curr.append(peaks[i])
        else:
            clusters.append(int(np.mean(curr)))
            curr = [peaks[i]]
    clusters.append(int(np.mean(curr)))
    
    # Filter very small clusters (noise)
    valid_clusters = [c for c in clusters if c > 5] 
    if not valid_clusters: 
        return None

    #LARGE_GAP_THRESHOLD = 25
    
    # STANDARD LOGIC: Jump over the Outer Frame
    for i in range(1, len(valid_clusters)):
        current_line = valid_clusters[i]
        prev_line = valid_clusters[i-1]
        
        dist_from_start = current_line - valid_clusters[0]
        if dist_from_start > max_search_dist:
            break
            
        gap = current_line - prev_line
        
        #if gap > LARGE_GAP_THRESHOLD:
        if gap > large_gap_threshold:
            return current_line
    
    return valid_clusters[-1] if valid_clusters else None

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
    
    if len(clean_pts) < 2: return None
    
    if orientation == 'h':
        m, c = np.polyfit(clean_pts[:, 0], clean_pts[:, 1], 1)
    else:
        m, c = np.polyfit(clean_pts[:, 1], clean_pts[:, 0], 1) 
        
    return m, c

def intersect(lh, lv):
    if not lh or not lv: return (0,0)
    m1, c1 = lh
    m2, c2 = lv
    
    det = 1 - m1*m2
    if abs(det) < 1e-5: return (0,0) 
    
    x = (m2 * c1 + c2) / det
    y = m1 * x + c1
    return (x, y)


def detect_frame_projection(image_path, world_coords, expected_ppm):
    
    img = read_image_gray_any(image_path)

    h, w = img.shape
    
    # --- SCAN DEPTHS ---
    margin_x = int(w * 0.15)
    margin_x_right = int(w * 0.20)
    margin_y = int(h * 0.10)
    margin_y_bottom = int(h * 0.30)
    
    # --- SEARCH LIMITS ---
    limit_top = int(h * 0.025)   
    limit_left = int(w * 0.05)   
    limit_right = int(w * 0.08)   
    limit_bot = int(h * 0.12)     
    
    # --- CLEANING KERNELS (relative to image size) ---
    clean_kernel_h_strong = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, w // 120), 1)
    )
    clean_kernel_v_strong = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(15, h // 120))
    )
    clean_kernel_v_weak = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, h // 500))
    )

    # Adaptive strip count (height-dependent)
    strips = max(20, h // 200)
    cw, ch = w // strips, h // strips
    
    top_pts, bot_pts, left_pts, right_pts = [], [], [], []
    
    # ========== TOP ==========
    for i in range(strips):
        strip_t = img[0:margin_y, i*cw:(i+1)*cw]
        y = find_line_in_strip_projection(strip_t, 'h', limit_top)
        if y is not None: 
            top_pts.append((i*cw + cw//2, y, i))  # (x, y, strip_index)

    # ========== BOTTOM ==========
    for i in range(strips):
        if i < 5 or i >= 15:
            raw_strip = img[h-margin_y_bottom:h, i*cw:(i+1)*cw]
            
            # Strong Clean (Remove Dashed Lines)
            inv = cv2.bitwise_not(raw_strip)
            cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, clean_kernel_h_strong)
            strip_b_clean = cv2.bitwise_not(cleaned)
            strip_b = np.flipud(strip_b_clean) 
            
            # Mask Legend
            mask_zone_size = int(h * 0.055) 
            if mask_zone_size < strip_b.shape[0]:
                strip_b[0:mask_zone_size, :] = 255 
            
            y_loc = find_line_in_strip_projection(strip_b, 'h', limit_bot)
            if y_loc is not None: 
                bot_pts.append((i*cw + cw//2, h - 1 - y_loc, i))  # (x, y, strip_index)

    # ========== LEFT ==========
    for i in range(strips):
        raw_strip_l = img[i*ch:(i+1)*ch, 0:margin_x]
        
        # Strong Clean (Remove Dashed Lines)
        inv_l = cv2.bitwise_not(raw_strip_l)
        cleaned_l = cv2.morphologyEx(inv_l, cv2.MORPH_OPEN, clean_kernel_v_strong)
        strip_l_clean = cv2.bitwise_not(cleaned_l)
        
        x = find_line_in_strip_projection(strip_l_clean, 'v', limit_left)
        if x is not None: 
            left_pts.append((x, i*ch + ch//2, i))  # (x, y, strip_index)

    # ========== RIGHT ==========
    for i in range(strips):
        raw_strip_r = img[i*ch:(i+1)*ch, w-margin_x_right:w]
        
        # Weak Clean (Keep frame safe)
        inv_r = cv2.bitwise_not(raw_strip_r)
        cleaned_r = cv2.morphologyEx(inv_r, cv2.MORPH_OPEN, clean_kernel_v_weak)
        strip_r_clean = cv2.bitwise_not(cleaned_r)
        strip_r = np.fliplr(strip_r_clean) 
        
        # Mask Outer Frame
        mask_zone_size_r = int(w * 0.035) 
        if mask_zone_size_r < strip_r.shape[1]:
            strip_r[:, 0:mask_zone_size_r] = 255 
            
        x_loc = find_line_in_strip_projection(strip_r, 'v', limit_right)
        if x_loc is not None:
            right_pts.append((w - 1 - x_loc, i*ch + ch//2, i))  # (x, y, strip_index)

    # ========== LINE FITTING ==========
    residual_thresh = max(8.0, 0.0015 * max(h, w))

    # Weighted fits (using strip position)
    lt_simple = fit_line_weighted(top_pts, 'h', strips)
    lb_simple = fit_line_weighted(bot_pts, 'h', strips)
    ll_simple = fit_line_weighted(left_pts, 'v', strips)
    lr_simple = fit_line_weighted(right_pts, 'v', strips)

    # Robust fits (existing RANSAC method - also uses 3-tuple points)
    lt_rob = robust_fit_line(top_pts, 'h', residual_thresh)
    lb_rob = robust_fit_line(bot_pts, 'h', residual_thresh)
    ll_rob = robust_fit_line(left_pts, 'v', residual_thresh)
    lr_rob = robust_fit_line(right_pts, 'v', residual_thresh)

    # ========== CANDIDATE SELECTION ==========
    candidates = []

    # Candidate 1: weighted simple
    if all([lt_simple, lb_simple, ll_simple, lr_simple]):
        px_simple = [
            intersect(lt_simple, ll_simple),
            intersect(lt_simple, lr_simple),
            intersect(lb_simple, lr_simple),
            intersect(lb_simple, ll_simple),
        ]
        score_simple = score_candidate(px_simple, world_coords, w, h, expected_ppm)
        candidates.append(("simple", px_simple, score_simple))

    # Candidate 2: robust
    if all([lt_rob, lb_rob, ll_rob, lr_rob]):
        px_rob = [
            intersect(lt_rob, ll_rob),
            intersect(lt_rob, lr_rob),
            intersect(lb_rob, lr_rob),
            intersect(lb_rob, ll_rob),
        ]
        score_rob = score_candidate(px_rob, world_coords, w, h, expected_ppm)
        candidates.append(("robust", px_rob, score_rob))

    if not candidates:
        raise ValueError("Detection failed (no valid rectangle candidates)")

    # Choose best candidate
    max_allowed_shift = 0.001 * max(w, h)
    shift = None

    if all([lt_simple, lb_simple, ll_simple, lr_simple]) and all([lt_rob, lb_rob, ll_rob, lr_rob]):
        shift = mean_corner_distance(px_simple, px_rob)

    if shift is not None and shift > max_allowed_shift and score_rob >= score_simple:
        best_name, pixel_coords, best_score = "simple", px_simple, score_simple
    else:
        best_name, pixel_coords, best_score = min(
            candidates,
            key=lambda x: x[2],
        )
    
    # Extract just (x, y) for debug visualization
    debug_top_pts = [(p[0], p[1]) for p in top_pts]
    debug_bot_pts = [(p[0], p[1]) for p in bot_pts]
    debug_left_pts = [(p[0], p[1]) for p in left_pts]
    debug_right_pts = [(p[0], p[1]) for p in right_pts]
    
    debug_data = {
        "top_pts": debug_top_pts,
        "bot_pts": debug_bot_pts,
        "left_pts": debug_left_pts,
        "right_pts": debug_right_pts,
        "best_candidate": best_name,
        "best_score": float(best_score),
    }

    return pixel_coords, debug_data

# ==========================================
# PART 3: PROCESSOR & ANALYTICS
# ==========================================

def calculate_stats(pixel_coords, world_coords):
    """
    Calculates RMSE, Scale (PPM), and Aspect Ratio consistency.
    """
    pts_px = np.array(pixel_coords, dtype=np.float32)
    pts_wld = np.array(world_coords, dtype=np.float32)
    
    # 1. RMSE (Geometric Distortion) - Affine Fit
    A = np.zeros((8, 6))
    b = np.zeros((8))
    for i in range(4):
        px, py = pts_px[i]; wx, wy = pts_wld[i]
        A[2*i] = [px, py, 1, 0, 0, 0]; b[2*i] = wx
        A[2*i+1] = [0, 0, 0, px, py, 1]; b[2*i+1] = wy
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    sum_sq_err = 0
    for i in range(4):
        px, py = pts_px[i]; wx_true, wy_true = pts_wld[i]
        wx_pred = x[0]*px + x[1]*py + x[2]
        wy_pred = x[3]*px + x[4]*py + x[5]
        sum_sq_err += (wx_pred - wx_true)**2 + (wy_pred - wy_true)**2
    rmse = math.sqrt(sum_sq_err / 4.0)

    # 2. DIMENSIONS & RATIOS
    # Pixels
    px_w_top = math.hypot(pts_px[1][0]-pts_px[0][0], pts_px[1][1]-pts_px[0][1])
    px_w_bot = math.hypot(pts_px[2][0]-pts_px[3][0], pts_px[2][1]-pts_px[3][1])
    px_h_left = math.hypot(pts_px[3][0]-pts_px[0][0], pts_px[3][1]-pts_px[0][1])
    px_h_right = math.hypot(pts_px[2][0]-pts_px[1][0], pts_px[2][1]-pts_px[1][1])
    
    px_width_avg = (px_w_top + px_w_bot) / 2.0
    px_height_avg = (px_h_left + px_h_right) / 2.0
    px_ar = px_width_avg / px_height_avg if px_height_avg > 0 else 0

    # World (GeoJSON)
    wld_w_top = math.hypot(pts_wld[1][0]-pts_wld[0][0], pts_wld[1][1]-pts_wld[0][1])
    wld_w_bot = math.hypot(pts_wld[2][0]-pts_wld[3][0], pts_wld[2][1]-pts_wld[3][1])
    wld_h_left = math.hypot(pts_wld[3][0]-pts_wld[0][0], pts_wld[3][1]-pts_wld[0][1])
    wld_h_right = math.hypot(pts_wld[2][0]-pts_wld[1][0], pts_wld[2][1]-pts_wld[1][1])
    
    wld_width_avg = (wld_w_top + wld_w_bot) / 2.0
    wld_height_avg = (wld_h_left + wld_h_right) / 2.0
    wld_ar = wld_width_avg / wld_height_avg if wld_height_avg > 0 else 0

    # 3. METRICS
    ppm = px_width_avg / wld_width_avg if wld_width_avg > 0 else 0
    ar_diff = abs(px_ar - wld_ar) / wld_ar if wld_ar > 0 else 1.0

    return rmse, ppm, px_width_avg, ar_diff

def score_candidate(pixel_coords, world_coords, img_w, img_h, expected_ppm=None):
    rmse, ppm, _, ar_diff = calculate_stats(pixel_coords, world_coords)

    pts = np.asarray(pixel_coords, dtype=np.float32)

    # Penalize corners outside image
    oob_penalty = 0.0
    if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= img_w):
        oob_penalty += 1000.0
    if np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= img_h):
        oob_penalty += 1000.0

    ppm_penalty = 0.0
    if expected_ppm is not None and expected_ppm > 0:
        ppm_penalty = abs(ppm - expected_ppm) / expected_ppm

    return rmse + 50.0 * ar_diff + 20.0 * ppm_penalty + oob_penalty

def process_image(img_path, geo_info, epsg, output_dir, write_warps=False):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    ml_out_path = os.path.join(output_dir, base_name + "_georef.tif")

    result = {
        "frame_detected": False,
        "debug_written": False,
        "warp_written": False,
        "quality_ok": False,
        "message": "",
        "rmse": 0.0,
        "ppm": 0.0,
        "px_width": 0.0,
        "ar_diff": 0.0,
        "best_candidate": "",
    }

    try:
        world_coords = geo_info["coords"]

        # 1. Detection
        pixel_coords, debug_data = detect_frame_projection(
            img_path,
            world_coords=world_coords,
            expected_ppm=0.4724,
        )
        result["frame_detected"] = True
        result["best_candidate"] = debug_data.get("best_candidate", "")

        # 2. Stats
        rmse, ppm, px_w, ar_diff = calculate_stats(pixel_coords, world_coords)
        result["rmse"] = rmse
        result["ppm"] = ppm
        result["px_width"] = px_w
        result["ar_diff"] = ar_diff

        # 3. Debug image
        debug_out_path = os.path.join(output_dir, base_name + "_debug.png")
        save_debug_overlay(
            image_path=img_path,
            pixel_coords=pixel_coords,
            top_pts=debug_data["top_pts"],
            bot_pts=debug_data["bot_pts"],
            left_pts=debug_data["left_pts"],
            right_pts=debug_data["right_pts"],
            out_path=debug_out_path,
            debug_data=debug_data,
        )
        result["debug_written"] = True

        # 4. Optional warp
        if write_warps:
            src_data = read_image_color_any(img_path)
            h, w, _ = src_data.shape

            mem_drv = gdal.GetDriverByName("MEM")
            ds = mem_drv.Create("", w, h, 3, gdal.GDT_Byte)
            for i in range(3):
                ds.GetRasterBand(i + 1).WriteArray(src_data[:, :, i])

            gdal_gcps = [
                gdal.GCP(
                    world_coords[i][0],
                    world_coords[i][1],
                    0,
                    pixel_coords[i][0],
                    pixel_coords[i][1],
                )
                for i in range(4)
            ]

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            vrt = gdal.Translate(
                "",
                ds,
                format="VRT",
                outputSRS=srs.ExportToWkt(),
                GCPs=gdal_gcps,
            )

            warp_ds = gdal.Warp(
                ml_out_path,
                vrt,
                dstSRS=srs,
                polynomialOrder=1,
                tps=False,
                resampleAlg=gdal.GRA_Lanczos,
                format="COG",
                dstAlpha=True,
                creationOptions=[
                    "COMPRESS=LZW",
                    "PREDICTOR=2",
                    "BIGTIFF=IF_NEEDED",
                    "OVERVIEWS=IGNORE_EXISTING",
                ],
            )
            result["warp_written"] = warp_ds is not None
        else:
            result["warp_written"] = False

        # 5. Quality
        # Check geometry quality and whether we are applying warps, to include quality when debugging.
        geometry_ok = (
            rmse <= 20.0
            and ar_diff <= 0.02
            and 0.460 <= ppm <= 0.485
        )

        result["quality_ok"] = geometry_ok and (result["warp_written"] if write_warps else True)

        result["message"] = "Success"
        return result

    except Exception as e:
        result["message"] = str(e)
        return result

def create_legend_file(folder_path):
    legend_text = """
================================================================================
                           GEOREFERENCING DIAGNOSTICS KEY
================================================================================

1. METRIC DEFINITIONS & CALCULATIONS
====================================

[RMSE_Meters] : Root Mean Square Error (Geometric Distortion)
-------------------------------------------------------------
Definition: Measures how much the 4 detected corners deviate from a perfect 
            rectangle when fitted to the world coordinates using an Affine transform.
Calculation: √((∑(x_calc - x_true)² + ∑(y_calc - y_true)²) / 4)
Interpretation:
  * 0.0 - 5.0m  : Excellent. The detected shape is a perfect rectangle.
  * 10.0 - 20.0m: Warning. Paper shrinkage or slight fold.
  * > 20.0m     : FAILURE. The shape is "Twisted". 
                  (e.g., Top-Left is correct, but Bottom-Right snapped to a text label).

[PPM_Scale] : Pixels Per Meter (Scale Consistency Check)
--------------------------------------------------------
Definition: Validates if the detected frame width (in pixels) matches the 
            known map scale and scan resolution.
Calculation for 1:25,000 Map at 300 DPI:
  * 1 Inch = 0.0254 Meters
  * Scan Resolution = 300 Pixels / Inch
  * Real World Scale = 25,000
  * Formula: (300 / 0.0254) / 25,000 = 0.4724 Pixels per Meter
Target:
  * Ideally ~0.472.
  * We allow a small tolerance (+/- 3%) for paper shrinkage/scanning artifacts.

[AR_Diff_Percent] : Aspect Ratio Difference
-------------------------------------------
Definition: Compares the proportions (Width / Height) of the detected image box 
            vs. the official GeoJSON coordinates.
Calculation: |(Ratio_Image - Ratio_World) / Ratio_World| * 100
Interpretation:
  * 0.0% : The shapes are identical proportions.
  * > 2.0% : The detected box is too tall or too wide compared to reality.
             (Usually means the detection included the Legend or Title).


2. DIAGNOSIS CODES EXPLAINED
============================

[OK]
-------------------------------------------------------------
The detection is geometrically sound. The shape is rectangular (RMSE low), 
the size matches 300 DPI 1:25k (PPM ~0.47), and proportions are correct.

[BAD GEOMETRY (Twisted)]
-------------------------------------------------------------
Condition: RMSE > 20.0 meters.
Physical Meaning: The 4 points found do not form a rectangle.
Likely Cause: Three corners were detected correctly, but the fourth corner 
              snapped to a stray mark, stamp, or text inside/outside the map.

[WRONG SHAPE (Trapezoid/Wrong Ratio)]
-------------------------------------------------------------
Condition: Aspect Ratio Difference > 2%.
Physical Meaning: The box is a rectangle, but it is too tall (usually).
Likely Cause: The bottom line detection snapped to the bottom of the 
              Legend Box instead of the Map Frame.

[OUTER FRAME (Too Big)]
-------------------------------------------------------------
Condition: PPM > 0.485 (Image is > 3% wider than expected).
Physical Meaning: The detected width in pixels is too large for the known meters.
Likely Cause: The code ignored the thin Inner Frame and snapped to the 
              thick decorative Outer Frame.

[INNER GRID (Too Small)]
-------------------------------------------------------------
Condition: PPM < 0.460 (Image is < 3% narrower than expected).
Physical Meaning: The detected width in pixels is too small.
Likely Cause: The code jumped over the white margin and snapped to the 
              first Coordinate Grid Line inside the map content.

[CHECK VISUALLY]
-------------------------------------------------------------
Condition: RMSE between 10.0m and 20.0m.
Physical Meaning: The map is technically a rectangle and the scale is correct, 
                  but there is higher-than-average distortion.
Likely Cause: Severe paper warping, folding, or the map was not scanned perfectly flat.

[MISSING_FROM_FOLDER]
-------------------------------------------------------------
Condition: The filename appears in the GeoJSON database but not in your folder.
Note: This check is smart—it only reports missing maps for the specific 
      1:100k regions (e.g., K-34-70) found in your input folder.
================================================================================
"""
    with open(os.path.join(folder_path, "_DIAGNOSIS_KEY.txt"), "w", encoding='utf-8') as f:
        f.write(legend_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Georeference scanned maps using frame detection and optional GDAL warp."
    )
    parser.add_argument(
        "--write-warps",
        action="store_true",
        help="Run the slow GDAL warp step and write georeferenced COG outputs.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Write warps: {'YES' if args.write_warps else 'NO'}")

    if not os.path.exists(OUTPUT_FOLDER): 
        os.makedirs(OUTPUT_FOLDER)
    
    create_legend_file(OUTPUT_FOLDER)
    
    # DYNAMIC REPORT NAME
    geojson_stem = GEOJSON_PATH.stem.replace('_grid', '')
    report_file = os.path.join(OUTPUT_FOLDER, f"{geojson_stem}_georef_report.csv")
    
    db, epsg = load_geojson_database(GEOJSON_PATH, GEOJSON_NAME_FIELD)
    
    found_map_names = set()
    active_parents = set()

    files = []
    if not os.path.exists(INPUT_FOLDER):
        print("Input folder not found.")
        return
        
    # search for the relevant image formats recursively
    # including into subfolders to find all maps in active regions
    for t in ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF'): 
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', t), recursive=True))

    print(f"Found {len(files)} images.")

    matched_count = 0
    frame_detected_count = 0
    debug_written_count = 0
    warp_written_count = 0
    quality_ok_count = 0
    skipped_count = 0
    
    with open(report_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename",
            "Original_Name",
            "Matched_Name",
            "Frame_Detected",
            "Debug_Written",
            "Warp_Written",
            "Quality_OK",
            "RMSE_m",
            "PPM_Scale",
            "AR_Diff_Percent",
            "Diagnosis",
            "Best_Candidate",
            "Message",
        ])
        
        # --- PROCESS IMAGES with Progress Bar ---
        for img_path in tqdm(files, desc="Processing Maps", unit="map"):
            filename = os.path.basename(img_path)
            keys = generate_match_keys(os.path.splitext(filename)[0])
            match = None
            for key in keys:
                if key in db: match = db[key]; break
            
            if match:
                orig_name = match['original_name']
                found_map_names.add(orig_name)
                
                # EXTRACT 1:100k PARENT (Grouping Logic)
                parts = orig_name.split('-')
                if len(parts) >= 3:
                    parent_id = "-".join(parts[:3])
                    active_parents.add(parent_id)
                
                result = process_image(
                    img_path=img_path, 
                    geo_info=match,
                    epsg=epsg, 
                    output_dir=OUTPUT_FOLDER,
                    write_warps=args.write_warps
                )

                matched_count += 1

                if result["frame_detected"]:
                    frame_detected_count += 1
                if result["debug_written"]:
                    debug_written_count += 1
                if result["warp_written"]:
                    warp_written_count += 1
                if result["quality_ok"]:
                    quality_ok_count += 1
                
                diagnosis = "OK"
                if not result["frame_detected"]:
                    diagnosis = "DETECTION_FAILED"
                elif not result["warp_written"]:
                    diagnosis = "WARP_FAILED"
                elif result["rmse"] > 20.0:
                    diagnosis = "BAD GEOMETRY (Twisted)"
                elif result["ar_diff"] > 0.02:
                    diagnosis = "WRONG SHAPE (Trapezoid/Wrong Ratio)"
                elif result["ppm"] > 0.485:
                    diagnosis = "OUTER FRAME (Too Big)"
                elif result["ppm"] < 0.460:
                    diagnosis = "INNER GRID (Too Small)"
                elif result["rmse"] > 10.0:
                    diagnosis = "CHECK VISUALLY"
                
                writer.writerow([
                    filename,
                    orig_name,
                    "YES",
                    "YES" if result["frame_detected"] else "NO",
                    "YES" if result["debug_written"] else "NO",
                    "YES" if result["warp_written"] else "NO",
                    "YES" if result["quality_ok"] else "NO",
                    f'{result["rmse"]:.2f}',
                    f'{result["ppm"]:.4f}',
                    f'{result["ar_diff"] * 100:.2f}',
                    diagnosis,
                    result["best_candidate"],
                    result["message"],
                ])
                f.flush()
            else:
                skipped_count += 1
                writer.writerow([
                    filename,
                    "NONE",
                    "NO",
                    "NO",
                    "NO",
                    "NO",
                    "NO",
                    "",
                    "",
                    "",
                    "NO MATCH",
                    "",
                    "",
                ])

        # --- FIND MISSING MAPS ---
        print("\nChecking for missing maps in active 1:100k regions...")
        
        all_db_names = set()
        for key, entry in db.items():
            name = entry['original_name']
            parts = name.split('-')
            if len(parts) >= 3:
                db_parent = "-".join(parts[:3])
                if db_parent in active_parents:
                    all_db_names.add(name)
            
        missing_maps = all_db_names - found_map_names
        missing_count = 0
        
        for name in sorted(missing_maps):
            writer.writerow(["(NO IMAGE)", name, "MISSING", "", "", "", "MISSING_FROM_FOLDER", f"Map missing from active region"])
            missing_count += 1
            
        print(f"\nProcessing complete. Report: {report_file}")
        print(f"Images Found:        {len(files)}")
        print(f"Matched to DB:       {matched_count}")
        print(f"Frame Detected:      {frame_detected_count}")
        print(f"Debug Written:       {debug_written_count}")
        print(f"Warp Written:        {warp_written_count}")
        print(f"Quality OK:          {quality_ok_count}")
        print(f"Skipped (No Match):  {skipped_count}")
        print(f"Maps Missing:        {missing_count}")
        print(f"Legend file created: {os.path.join(OUTPUT_FOLDER, '_DIAGNOSIS_KEY.txt')}")

if __name__ == "__main__":
    main()