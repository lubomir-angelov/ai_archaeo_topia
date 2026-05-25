# Georeferencing Improvement Plan

## Current Performance

**200 matched images, only 61 pass quality checks (30.5%).**

| Failure Mode | Count | % of Matched |
|---|---|---|
| AR_Diff > 2% (wrong shape) | 112 | 56% |
| PPM < 0.460 (too small / inner grid) | 31 | 15.5% |
| PPM > 0.485 (too big / outer frame) | 25 | 12.5% |
| RMSE > 20 (twisted geometry) | 9 | 4.5% |

**AR_Diff > 2% is the dominant failure at 56% of all images.** The detected frame box aspect ratio does not match the GeoJSON reference shape — the frame detection systematically produces boxes that are too tall or too wide.

---

## Root Cause Analysis

### 1. Asymmetric margins cause uneven strip sampling

`detect_frame_projection` uses different margins per side:

- `margin_x = 15%`, `margin_x_right = 20%`
- `margin_y = 10%`, `margin_y_bottom = 30%`

The bottom margin is 3x the top margin (to avoid the legend). This means the bottom frame detection operates on a much wider strip region than the top, and the strip-based sampling is uneven across sides.

### 2. Bottom strip legend masking is hardcoded

Lines 349-354 blank out the first `mask_zone_size = h * 0.055` pixels of the flipped bottom strip to skip the legend. But legend position varies between map sheets. The mask either:
- Misses the legend (detecting legend text/box edges as frame lines)
- Is too aggressive (losing valid frame data near the bottom edge)

### 3. Right strip outer-frame masking is hardcoded

Lines 398-403 set `mask_zone_size_r = w * 0.011` to skip the decorative outer frame on the right side. This is a hardcoded fraction that does not adapt to different map layouts.

### 4. Strip-based detection is noisy at edges

The weighted fitting (`fit_line_weighted` in `utils.py:328`) gives higher weight to edge strips (weights: `0.3 + 0.7 * distance_from_center`), which is correct in principle. But edge strips are also where the frame line is most likely to be broken, faded, or obscured by map content. When edge strips fail to detect, the weighted fit is pulled inward by center strips that may have latched onto grid lines or internal frame elements.

### 5. Gap-based line selection latches onto wrong lines

`find_line_in_strip_projection` (lines 234-246) jumps over the first large gap to find the inner frame. The gap threshold is `max(8, int(profile_len * gap_ratio))` with `gap_ratio=0.04`. A 1000-pixel-wide strip gives a 40-pixel gap threshold. If the inner and outer frames are closer than this, or if internal grid lines create false gaps, the function latches onto the wrong line.

### 6. Sides are detected independently with no cross-validation

Each of the four sides (top, bottom, left, right) is detected in isolation. There is no check during detection whether the resulting lines are approximately parallel/perpendicular. If three sides are correct but one is off by even 50px, the AR_Diff explodes past 2%.

### 7. Why tweaking params breaks other sides

When you tighten `gap_ratio` or `threshold_ratio` to fix left/right failures, top/bottom strips (which have different margin sizes, different content, and different gap structures) lose their detections entirely or latch onto noise. The four sides have fundamentally different profiles but share the same detection parameters.

---

## Proposed Improvements

### P1: Cross-side consistency check during candidate selection

**Impact: High | Effort: Low**

Currently (lines 431-472), the code generates two candidates (simple/robust) and picks the one with the lower score. Add a check whether the detected lines are approximately parallel/perpendicular:

- Are top and bottom lines roughly parallel (similar slope)?
- Are left and right lines roughly parallel?
- Are adjacent lines roughly perpendicular?

If one side's line is not parallel to its counterpart, flag it as a likely mis-detection and either discard that candidate or penalize it heavily.

### P1: Make legend/outer-frame masking adaptive per-image

**Impact: High | Effort: Medium**

Instead of hardcoded fractions (`h * 0.055`, `w * 0.011`), determine mask zones per-image by analyzing the projection profile:

- For the bottom strip: find the first strong horizontal line in the bottom 30% that is NOT at the expected frame distance — that's likely the legend. Set the mask to extend slightly past it.
- For the right strip: detect whether there are two strong vertical lines close together (inner frame + outer frame). Set the mask to skip only the outer one.

### P2: Add consensus detection mode

**Impact: Medium | Effort: Medium**

Run strip detection with multiple `gap_ratio` values (e.g., 0.02, 0.04, 0.06) and take the mode/median position for each side. This handles cases where the gap-based logic latches onto the wrong line on some strips but not others.

### P2: Relax AR_Diff threshold with scale-aware validation

**Impact: Medium | Effort: Low**

The 2% AR_Diff threshold (line 670) may be too strict for scanned historical maps with non-uniform paper shrinkage. Maps shrink differently in X vs Y. A better approach:

- Compare detected aspect ratio against expected shrinkage pattern derived from PPM.
- If PPM is consistent on top/bottom but different on left/right, that's directional shrinkage, not a detection error.
- Consider separate thresholds for width-ratio vs height-ratio mismatches.

### P3: Rebalance the scoring function

**Impact: Medium | Effort: Low**

Line 562: `return rmse + 50.0 * ar_diff + 20.0 * ppm_penalty + oob_penalty`

The AR_Diff multiplier (50x) dominates the score. This means candidate selection primarily optimizes for aspect ratio match, not geometric correctness (RMSE). If the true frame has slight distortion (common with old paper maps), the scorer prefers a geometrically worse candidate that happens to have the right aspect ratio.

Proposed changes:
- Reduce AR_Diff weight (e.g., 20x instead of 50x)
- Add a separate "geometry quality" gate that rejects candidates with twisted corners regardless of AR
- Consider adding a per-side confidence term based on point density

### P3: Tune per-side detection parameters

**Impact: Medium | Effort: Low**

The left/right sides already use different morphological kernels (`clean_kernel_v_weak` vs `clean_kernel_v_strong`), but top/bottom do not. Consider:

- Using a stronger kernel on top (where frame lines may be thinner/cleaner) and weaker on bottom (where the frame may be thicker or partially obscured by legend bleed).
- Using different `gap_ratio` values per side: tighter for left/right (where frames are typically clean vertical lines), looser for top/bottom (where the frame may have gaps from title text or map content).

### P3: PPM < 0.460 failures suggest systematic inward snapping

31 images have PPM < 0.460 ("INNER GRID" diagnosis) — the detection jumped past the inner frame to a coordinate grid line. The `gap_ratio` and `max_search_dist` parameters should be tuned per-side. The inner frame is typically the strongest continuous line near the edge; coordinate grid lines are usually dashed or shorter. The morphological opening kernel could be sized to preferentially remove dashed grid lines while preserving solid frame lines on all four sides.

---

## Quick Wins (No Code Changes)

1. **Adjust PPM tolerance**: The current range `0.460 - 0.485` (±3% from 0.4724) may be too narrow. Historical maps with paper shrinkage could legitimately fall outside this range. Consider widening to `0.455 - 0.490` and relying more on AR_Diff + RMSE for quality gating.

2. **Tighten AR_Diff only when PPM is also off**: If PPM is in range but AR_Diff is high, that's a genuine detection error. If PPM is also off, the AR_Diff failure may be a symptom (e.g., outer frame detected on one side and inner on the other) rather than the root cause.

3. **Review debug overlays for the 112 AR_Diff failures**: The `_debug.png` files show detected points per side. A quick visual scan of the worst offenders will likely reveal a pattern (e.g., "always the bottom side snaps to the legend box on maps with large title blocks").
