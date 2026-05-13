#%%
from pyrasa.irasa import irasa
import numpy as np
import pandas as pd
from eyetools.annotateblinks import add_blinkvec2raw

from scipy.signal import savgol_filter
from scipy.stats import skew, entropy

#%%


def ocular_activity_measures(
    x,
    y,
    sampling_rate,
    remove_nan=True,
    savgol_window=21,
    savgol_polyorder=2,
):
    """
    Compute summary measures of general ocular activity.

    Parameters
    ----------
    x, y : array-like
        Horizontal and vertical gaze position signals.
        Units can be pixels, degrees visual angle, etc.
    sampling_rate : float
        Sampling rate in Hz.
    remove_nan : bool
        If True, removes samples with NaNs.

    Returns
    -------
    dict
        Dictionary containing summary measures of ocular activity.

    Metrics
    -------
    rms_displacement : float
        Root-mean-square (RMS) displacement between consecutive gaze samples.
        Reflects the typical magnitude of moment-to-moment eye movements.
        Higher values indicate larger or more unstable gaze shifts.

    velocity : array
    Velocity is computed as the Euclidean displacement between consecutive
    gaze samples multiplied by the sampling rate:

    velocity = displacement * sampling_rate

    where displacement is defined as:

    displacement = sqrt(dx^2 + dy^2)

    This yields the movement speed of the gaze signal in units per second
    (e.g., pixels/s or degrees/s depending on the coordinate system).

    mean_velocity : float
        Mean gaze velocity across the recording.
        Computed from Euclidean displacement multiplied by the sampling rate.
        Sensitive to overall ocular activity level.

    median_velocity : float
        Median gaze velocity.
        More robust to transient high-velocity artifacts or saccades than the mean.

    sd_velocity : float
        Standard deviation of gaze velocity.
        Quantifies variability in ocular motion dynamics.

    p95_velocity : float
        95th percentile of gaze velocity.
        Captures high-velocity events such as rapid saccades while remaining
        less sensitive to extreme outliers than the maximum.

    total_path_length : float
        Total cumulative gaze displacement over time.
        Calculated as the sum of all sample-to-sample Euclidean displacements.
        Reflects the overall amount of eye movement during the recording.

    dispersion_x : float
        Standard deviation of horizontal gaze position.
        Quantifies spread of gaze along the x-axis.

    dispersion_y : float
        Standard deviation of vertical gaze position.
        Quantifies spread of gaze along the y-axis.

    radial_dispersion : float
        Mean radial distance from the gaze centroid.
        Provides a global measure of gaze dispersion independent of axis orientation.

    bcea : float
        Bivariate Contour Ellipse Area (BCEA), an estimate of the spatial area
        encompassing approximately 68% of gaze samples.
        Commonly used as a measure of fixation stability and spatial gaze variability.
        Lower BCEA values indicate more stable fixation behavior.

    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if remove_nan:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid samples.")

    # ------------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------------
    x_s = savgol_filter(x, savgol_window, savgol_polyorder)
    y_s = savgol_filter(y, savgol_window, savgol_polyorder)

    #dx = np.diff(x_s)
    #dy = np.diff(y_s)

    # 5 point velocity estimation; this is where the magic happens
    dt = 1 / sampling_rate
    dx = np.zeros_like(x_s)
    dy = np.zeros_like(y_s)
    dx[2:-2] = (x_s[4:] + x_s[3:-1] - x_s[1:-3] - x_s[:-4]) / (6*dt)
    dy[2:-2] = (y_s[4:] + y_s[3:-1] - y_s[1:-3] - y_s[:-4]) / (6*dt)

    # Euclidean displacement between samples
    velocity = np.sqrt(dx**2 + dy**2)

    # ------------------------------------------------------------------
    # Velocity statistics
    # ------------------------------------------------------------------
    mean_velocity = np.mean(velocity)
    median_velocity = np.median(velocity)
    sd_velocity = np.std(velocity)
    p95_velocity = np.percentile(velocity, 95)

    # ------------------------------------------------------------------
    # Gaze dispersion measures
    # ------------------------------------------------------------------

    # Simple SD-based dispersion
    dispersion_x = np.std(x_s)
    dispersion_y = np.std(y_s)

    # Radial dispersion
    center_x = np.mean(x_s)
    center_y = np.mean(y_s)

    radial_distance = np.sqrt((x_s - center_x)**2 + (y_s - center_y)**2)
    radial_dispersion = np.mean(radial_distance)

    # ------------------------------------------------------------------
    # BCEA (Bivariate Contour Ellipse Area)
    # Approximate 68% ellipse
    # ------------------------------------------------------------------
    sigma_x = np.std(x_s)
    sigma_y = np.std(y_s)
    rho = np.corrcoef(x_s, y_s)[0, 1]

    # 68% BCEA constant
    k = 1.146

    bcea = (
        2
        * np.pi
        * k
        * sigma_x
        * sigma_y
        * np.sqrt(1 - rho**2)
    )

    return {
        "velocity": velocity,
        "mean_velocity": mean_velocity,
        "median_velocity": median_velocity,
        "sd_velocity": sd_velocity,
        "p95_velocity": p95_velocity,
        "dispersion_x": dispersion_x,
        "dispersion_y": dispersion_y,
        "radial_dispersion": radial_dispersion,
        "bcea": bcea,
    }

#%%
def detect_velocity_peaks(
    velocity,
    fs,
    threshold_factor=6,
    min_duration_ms=10,
    merge_gap_ms=5,
):
    """
    Detect transient velocity events above background noise.

    Uses a robust threshold, detects suprathreshold segments,
    merges nearby segments, and extracts one peak per event.

    Parameters
    ----------
    velocity : array-like
        Velocity signal.
    fs : float
        Sampling rate in Hz.
    threshold_factor : float
        Multiplier for robust noise estimate.
    min_duration_ms : float
        Minimum event duration in milliseconds.
    merge_gap_ms : float
        Maximum gap between suprathreshold segments that will still
        be treated as one event.

    Returns
    -------
    peaks : ndarray
        Indices of detected peak maxima.
    threshold : float
        Detection threshold.
    above_threshold : ndarray
        Boolean mask of suprathreshold samples.
    segments : list of tuple
        Final merged event segments as (start, end) sample indices.
    """

    velocity = np.asarray(velocity, dtype=float)

    # Robust threshold
    med = np.nanmedian(velocity)
    mad = np.nanmedian(np.abs(velocity - med))
    robust_sd = 1.4826 * mad
    threshold = med + threshold_factor * robust_sd

    above_threshold = velocity > threshold

    # Find suprathreshold segments
    segments = []
    in_segment = False

    for i, val in enumerate(above_threshold):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(velocity)))

    # Merge nearby segments
    merge_gap_samples = int(merge_gap_ms / 1000 * fs)

    merged_segments = []

    if segments:
        current_start, current_end = segments[0]

        for next_start, next_end in segments[1:]:
            gap = next_start - current_end

            if gap <= merge_gap_samples:
                current_end = next_end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged_segments.append((current_start, current_end))

    # Keep only sufficiently long events and extract one peak per event
    min_duration_samples = int(min_duration_ms / 1000 * fs)

    final_segments = []
    peaks = []

    for start, end in merged_segments:
        duration = end - start

        if duration >= min_duration_samples:
            segment = velocity[start:end]
            peak_idx = start + np.nanargmax(segment)

            final_segments.append((start, end))
            peaks.append(peak_idx)

    peaks = np.asarray(peaks, dtype=int)

    return peaks, threshold, above_threshold, final_segments

#%%

def extract_eye_gazedata(
    rawVPixx,
    rawMEG,
    blinkthresh=50,
    duration=10,
    overlap=0.5,
    hset_info=(1, 2, 0.05),
    low_freq=0.5,
    high_freq=100,
    leftX="Left Eye x",
    leftY="Left Eye y",
    rightX="Right Eye x",
    rightY="Right Eye y",
    leftPupil="Left Eye Pupil Diameter",
    rightPupil="Right Eye Pupil Diameter",
):
    """
    Extract eye-tracking data from the eye with less signal loss,
    run IRASA, and extract blink information.
    """

    fs = int(round(rawVPixx.info["sfreq"]))

    left_loss = np.sum(rawVPixx.copy().pick(leftX).get_data())
    right_loss = np.sum(rawVPixx.copy().pick(rightX).get_data())

    if left_loss <= right_loss:
        chanX = leftX
        chanY = leftY
        chanPupil = leftPupil
        selected_eye = "left"
    else:
        chanX = rightX
        chanY = rightY
        chanPupil = rightPupil
        selected_eye = "right"

    raweyes = rawVPixx.copy().pick(
        [chanX, chanY, chanPupil]
    ).get_data()

    irasa_out = irasa(
        raweyes,
        fs=fs,
        band=(low_freq, high_freq),
        nperseg=duration * fs,
        noverlap=int(duration * fs * overlap),
        hset_info=hset_info,
    )

    _, blinks_df = add_blinkvec2raw(
        rawMEG,
        thresh=blinkthresh,
    )

    gazedata = {
        "selected_eye": selected_eye,
        "chanX": chanX,
        "chanY": chanY,
        "chanPupil": chanPupil,
        "fs": fs,
        "irasa_out": irasa_out,
        "blinks_df": blinks_df,
        "params": {
            "blinkthresh": blinkthresh,
            "duration": duration,
            "overlap": overlap,
            "hset_info": hset_info,
            "low_freq": low_freq,
            "high_freq": high_freq,
        },
    }

    return raweyes, gazedata

#%%

def compute_peak_statistics(
    peaks,
    velocity,
    fs,
    n_bins='auto',
):
    """
    Parameters
    ----------
    peaks : array-like
        Peak indices in sampling units.
    velocity : array-like
        Velocity signal.
    fs : float
        Sampling rate in Hz.
    n_bins : int
        Number of histogram bins for entropy estimation.

    Returns
    -------
    stats : dict
        Dictionary containing summary statistics of ocular velocity peaks.

    Outputs
    -------

    n_peaks : int
        Total number of detected ocular velocity events.

    peak_rate_hz : float
        Number of detected peaks per second.
        Reflects the overall frequency of transient ocular events such as
        microsaccades or corrective saccades.

    mean_peak_amp : float
        Mean peak amplitude of detected velocity events.

    median_peak_amp : float
        Median peak amplitude.
        Robust estimate of typical event magnitude.

    sd_peak_amp : float
        Standard deviation of peak amplitudes.
        Quantifies variability in event magnitude.

    p95_peak_amp : float
        95th percentile of peak amplitudes.
        Captures the upper range of ocular event magnitudes while remaining
        less sensitive to extreme outliers than the maximum.

    mean_ipi_sec : float
        Mean inter-peak interval (IPI) in seconds.
        Reflects the average temporal spacing between ocular events.

    median_ipi_sec : float
        Median inter-peak interval in seconds.
        Robust estimate of typical event spacing.

    sd_ipi_sec : float
        Standard deviation of inter-peak intervals.
        Quantifies variability in event timing.

    cv_ipi : float
        Coefficient of variation of inter-peak intervals:

            sd_ipi / mean_ipi

        Measures temporal irregularity of ocular events.

        Interpretation:
        - values near 1 indicate approximately random timing
        - values > 1 indicate bursty or clustered event structure
        - values < 1 indicate more regular timing

    skew_log_ipi : float
        Skewness of the log-transformed inter-peak interval distribution.
        Quantifies asymmetry in ocular event timing.

    ipi_entropy : float
        Shannon entropy of the log-inter-peak interval distribution.
        Higher values indicate broader and more heterogeneous temporal dynamics.

    peak_amplitudes : ndarray
        Array containing amplitudes of all detected peaks.

    ipi_sec : ndarray
        Array containing inter-peak intervals in seconds.

    log_ipi : ndarray
        Log10-transformed inter-peak intervals.
    """

    peaks = np.asarray(peaks)
    velocity = np.asarray(velocity)

    duration_sec = len(velocity) / fs

    # ---------------------------------------------------------
    # Peak rate
    # ---------------------------------------------------------

    peak_rate = len(peaks) / duration_sec

    # ---------------------------------------------------------
    # Peak amplitudes
    # ---------------------------------------------------------

    peak_amp = velocity[peaks]

    # ---------------------------------------------------------
    # Inter-peak intervals
    # ---------------------------------------------------------

    if len(peaks) > 1:

        ipi = np.diff(peaks) / fs
        log_ipi = np.log10(ipi)

        median_ipi = np.median(ipi)
        mean_ipi = np.mean(ipi)
        sd_ipi = np.std(ipi)

        cv_ipi = sd_ipi / mean_ipi

        skew_ipi = skew(log_ipi)

        if n_bins == "auto":
            n_bins = max(5, min(20, int(np.sqrt(len(log_ipi)))))

        # Entropy of log-IPI distribution
        hist, _ = np.histogram(
            log_ipi,
            bins=n_bins,
            density=True,
        )

        ipi_entropy = entropy(hist + 1e-12)

    else:

        ipi = np.array([])
        log_ipi = np.array([])

        median_ipi = np.nan
        mean_ipi = np.nan
        sd_ipi = np.nan
        cv_ipi = np.nan
        skew_ipi = np.nan
        ipi_entropy = np.nan

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    stats = {

        # Event density
        "n_peaks": len(peaks),
        "peak_rate_hz": peak_rate,

        # Peak amplitude
        "mean_peak_amp": np.mean(peak_amp),
        "median_peak_amp": np.median(peak_amp),
        "sd_peak_amp": np.std(peak_amp),
        "p95_peak_amp": np.percentile(peak_amp, 95),

        # Inter-peak intervals
        "mean_ipi_sec": mean_ipi,
        "median_ipi_sec": median_ipi,
        "sd_ipi_sec": sd_ipi,
        "cv_ipi": cv_ipi,

        # Distribution shape
        "skew_log_ipi": skew_ipi,
        "ipi_entropy": ipi_entropy,

        # Raw arrays
        "peak_amplitudes": peak_amp,
        "ipi_sec": ipi,
        "log_ipi": log_ipi,
    }

    return stats

#%%

def remove_blink_related_peaks(peaks, blinks_df, fs, pre_blink=0.15, post_blink=0.25):
    keep = np.ones(len(peaks), dtype=bool)

    for _, blink in blinks_df.iterrows():
        blink_start = int((blink["onset_sec"] - pre_blink) * fs)
        blink_end = int((blink["offset_sec"] + post_blink) * fs)

        in_blink_window = (peaks >= blink_start) & (peaks <= blink_end)
        keep[in_blink_window] = False

    return peaks[keep], keep

