from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Detection: white targets (large white calibration dots)
# ---------------------------------------------------------------------
def detect_white_targets(
    img_bgr: np.ndarray,
    min_area: float = 150,
    max_area: float = 5000,
    white_threshold: int = 220,
    blur_ksize: int = 5,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """
    Detect white calibration targets (large white dots).

    Returns
    -------
    centers : list of (cx, cy, area)
    bw      : binary mask used for detection
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img_gray, white_threshold, 255, cv2.THRESH_BINARY)

    if blur_ksize and blur_ksize > 1:
        bw = cv2.medianBlur(bw, blur_ksize)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy, area))

    centers = sorted(centers, key=lambda x: (x[1], x[0]))
    return centers, bw


# ---------------------------------------------------------------------
# Detection: measured fixation markers (rings)
#   - supports cyan/blue and magenta/pink (left vs right eye)
#   - suppresses trace dots via S/V thresholds + area + optional circularity
#   - optional "keep largest N" to enforce expected number of fixations
# ---------------------------------------------------------------------
def detect_measured_points(
    img_bgr: np.ndarray,
    hsv_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    min_area: float = 40,
    max_area: float = 20000,
    blur_ksize: int = 5,
    morph_open_iter: int = 1,
    expected_n: Optional[int] = 13,
    circularity_min: Optional[float] = 0.65,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """
    Detect measured fixation markers (ring-like colored markers).

    Parameters
    ----------
    hsv_ranges : list of (lowHSV, highHSV)
        Each low/high is (H,S,V). If None, uses robust defaults for rings.
    min_area : float
        Minimum contour area to be considered a fixation marker.
    expected_n : int or None
        If not None and more blobs are found, keeps the largest expected_n.
    circularity_min : float or None
        If not None, discards blobs with circularity < threshold.

    Returns
    -------
    centers : list of (cx, cy, area)
    mask : combined binary mask used for detection
    """
    # Defaults tuned for BRIGHT rings; reduces small/dark movement trace dots.
    if hsv_ranges is None:
        hsv_ranges = [
            # cyan/light blue rings
            ((70, 120, 120), (140, 255, 255)),
            # magenta/pink rings (upper hue band)
            ((140, 120, 120), (179, 255, 255)),
            # magenta/pink wrap-around near 0
            ((0, 120, 120), (12, 255, 255)),
        ]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in hsv_ranges:
        low_arr = np.array(low, dtype=np.uint8)
        high_arr = np.array(high, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low_arr, high_arr))

    if blur_ksize and blur_ksize > 1:
        mask = cv2.medianBlur(mask, blur_ksize)

    if morph_open_iter and morph_open_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        # Optional circularity filter
        if circularity_min is not None:
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = (4.0 * np.pi * area) / (perim * perim)
            if circ < circularity_min:
                continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        blobs.append((cx, cy, area))

    # Keep only the largest expected_n blobs if too many are detected
    if expected_n is not None and len(blobs) > expected_n:
        blobs = sorted(blobs, key=lambda x: x[2], reverse=True)[:expected_n]

    centers = sorted(blobs, key=lambda x: (x[1], x[0]))
    return centers, mask


# ---------------------------------------------------------------------
# Matching + summary
# ---------------------------------------------------------------------
def match_nearest(
    targets: List[Tuple[float, float, float]],
    measured: List[Tuple[float, float, float]],
    max_dist_px: float = 250,
) -> pd.DataFrame:
    """
    For each target, find the nearest measured point within max_dist_px.
    Each measured point is used at most once.
    """
    measured_xy = np.array([(m[0], m[1]) for m in measured], dtype=float)
    used = np.zeros(len(measured_xy), dtype=bool)

    rows = []
    for i, (tx, ty, _) in enumerate(targets):
        if len(measured_xy) == 0:
            rows.append((i, tx, ty, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        d = np.sqrt((measured_xy[:, 0] - tx) ** 2 + (measured_xy[:, 1] - ty) ** 2)
        d[used] = np.inf

        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > max_dist_px:
            rows.append((i, tx, ty, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        used[j] = True
        mx, my = measured_xy[j]
        dx = mx - tx
        dy = my - ty
        dist = float(np.sqrt(dx * dx + dy * dy))

        rows.append((i, tx, ty, mx, my, dx, dy, dist))

    df = pd.DataFrame(
        rows,
        columns=[
            "target_id",
            "target_x",
            "target_y",
            "measured_x",
            "measured_y",
            "dx_px",
            "dy_px",
            "dist_px",
        ],
    )
    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics from the match DataFrame.
    """
    valid = df["dist_px"].dropna()
    if len(valid) == 0:
        return {
            "n_targets": float(len(df)),
            "n_matched": 0.0,
            "median_dist_px": float("nan"),
            "rms_dist_px": float("nan"),
            "max_dist_px": float("nan"),
        }

    rms = float(np.sqrt(np.mean(valid**2)))
    return {
        "n_targets": float(len(df)),
        "n_matched": float(valid.shape[0]),
        "median_dist_px": float(np.median(valid)),
        "rms_dist_px": rms,
        "max_dist_px": float(np.max(valid)),
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def draw_overlay_opencv(img_bgr: np.ndarray, df: pd.DataFrame, out_path: Union[str, Path]) -> None:
    """
    Write an OpenCV overlay image showing targets, measured points, and match lines.
    """
    out = img_bgr.copy()

    for _, r in df.iterrows():
        tx, ty = int(r["target_x"]), int(r["target_y"])
        cv2.circle(out, (tx, ty), 12, (255, 255, 255), 2)

        if np.isfinite(r["measured_x"]):
            mx, my = int(r["measured_x"]), int(r["measured_y"])
            cv2.circle(out, (mx, my), 8, (255, 255, 0), 2)
            cv2.line(out, (tx, ty), (mx, my), (0, 255, 255), 2)

            dist = r["dist_px"]
            cv2.putText(
                out,
                f"{dist:.0f}px",
                (tx + 10, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def plot_polyresponse_results(
    img_bgr: np.ndarray,
    df: pd.DataFrame,
    title: Optional[str] = None,
    annotate: bool = False,
    ax=None,
):
    """
    Plot matching results on top of the screenshot using Matplotlib.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    ax.imshow(img_rgb)
    ax.set_axis_off()

    # Targets
    ax.scatter(
        df["target_x"],
        df["target_y"],
        s=160,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
        label="Targets",
    )

    matched = df[np.isfinite(df["measured_x"])]

    # Measured
    ax.scatter(
        matched["measured_x"],
        matched["measured_y"],
        s=90,
        facecolors="none",
        edgecolors="cyan",
        linewidths=2,
        label="Measured",
    )

    # Lines + optional annotations
    for _, r in matched.iterrows():
        ax.plot([r["target_x"], r["measured_x"]], [r["target_y"], r["measured_y"]], linewidth=1.5)
        if annotate:
            ax.text(r["target_x"] + 8, r["target_y"] - 8, f"{r['dist_px']:.0f}px", fontsize=8)

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="lower right")

    if created_fig:
        fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------
def analyze_vpixx_polyresponse(
    jpeg_path: Union[str, Path],
    # saving
    outdir: Optional[Union[str, Path]] = None,
    save_outputs: bool = False,
    make_overlay: bool = False,
    # matching
    max_dist_px: float = 250,
    # white detection
    white_min_area: float = 150,
    white_max_area: float = 5000,
    white_threshold: int = 220,
    # measured detection
    measured_hsv_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    measured_min_area: float = 40,
    measured_max_area: float = 20000,
    expected_n_measured: Optional[int] = 13,
    measured_circularity_min: Optional[float] = 0.65,
    # optional returns / diagnostics
    return_masks: bool = False,
    verbose: bool = False,
    # plotting
    plot: bool = False,
    annotate: bool = False,
    ax=None,
    save_plot_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, ...]:
    """
    Analyze a VPIXX PolyResponse screenshot (jpeg/png).

    Default return:
        (df, stats)

    If plot=True:
        (df, stats, (fig, ax))

    If return_masks=True:
        adds an extra dict {"white_mask":..., "measured_mask":...} at the end.
    """
    jpeg_path = Path(jpeg_path)

    # Prepare output dir
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(jpeg_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {jpeg_path}")

    # Detect targets
    targets, white_mask = detect_white_targets(
        img,
        min_area=white_min_area,
        max_area=white_max_area,
        white_threshold=white_threshold,
    )

    # Detect measured fixation markers
    measured, measured_mask = detect_measured_points(
        img,
        hsv_ranges=measured_hsv_ranges,
        min_area=measured_min_area,
        max_area=measured_max_area,
        expected_n=expected_n_measured,
        circularity_min=measured_circularity_min,
    )

    # Match and summarize
    df = match_nearest(targets, measured, max_dist_px=max_dist_px)
    stats = summarize(df)

    # Save outputs (CSV + masks + optional OpenCV overlay)
    if save_outputs and outdir is not None:
        offsets_path = outdir / f"{jpeg_path.stem}_offsets.csv"
        summary_path = outdir / f"{jpeg_path.stem}_summary.csv"
        df.to_csv(offsets_path, index=False)
        pd.Series(stats).to_csv(summary_path)

        cv2.imwrite(str(outdir / f"{jpeg_path.stem}_white_mask.png"), white_mask)
        cv2.imwrite(str(outdir / f"{jpeg_path.stem}_measured_mask.png"), measured_mask)

        if make_overlay:
            overlay_path = outdir / f"{jpeg_path.stem}_overlay.png"
            draw_overlay_opencv(img, df, overlay_path)

    if verbose:
        print("=== VPIXX PolyResponse Summary ===")
        print(f"Image: {jpeg_path}")
        print(f"Targets detected: {len(targets)}")
        print(f"Measured fixations detected: {len(measured)}")
        for k, v in stats.items():
            print(f"{k}: {v}")
        if save_outputs and outdir is not None:
            print(f"Saved to: {outdir.resolve()}")

    # Optional plot
    fig = ax_out = None
    if plot:
        title = (
            f"{jpeg_path.name} | "
            f"median={stats['median_dist_px']:.1f}px "
            f"RMS={stats['rms_dist_px']:.1f}px"
        )
        fig, ax_out = plot_polyresponse_results(img, df, title=title, annotate=annotate, ax=ax)

        if save_plot_path is not None:
            save_plot_path = Path(save_plot_path)
            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_plot_path), dpi=200)

    # Build return tuple
    ret = [df, stats]

    if plot:
        ret.append((fig, ax_out))

    if return_masks:
        ret.append({"white_mask": white_mask, "measured_mask": measured_mask})

    return tuple(ret)
