#%%
import mne
import numpy as np
from scipy.fftpack import hilbert
from scipy.ndimage import binary_opening, binary_closing
from scipy.signal import correlate, hilbert
from scipy.spatial.distance import cdist

# %%
def blinkfromMEG(raw, picks=['MISC010', 'MISC011'], threshold_percentile=99):
    '''
    Pass eye tracking channels from MEG raw data to extract blink binary vector.
    (here we label everything as “Blink“. in many cases it is simple data loss,
    that also occurs in VPixx mat-files as blinks)
    
    :param raw: raw MEG data including eye tracking channels
    :param picks: Labels of eye tracking channels in MEG data
    :param threshold_percentile: Percentile threshold to detect blinks
    :return: Binary vector indicating blink presence
    '''
    meg_blink = raw.get_data(picks=picks)
    meg_blink = np.mean(meg_blink, axis=0)
    meg_env = np.abs(hilbert(meg_blink))
    meg_blink_bin = meg_env > np.percentile(meg_env, threshold_percentile)
    return meg_blink_bin

#%% WHICH EYE MEASURED IN MEG??
def blinkfromVPixx(raw, pick='Left Eye Blink', threshold=0.5):
    '''
    Pass eye tracking channels from VPixx raw data to extract blink binary vector.
    (here we label everything as “Blink“. in many cases it is simple data loss,
    that also occurs in MEG as missing data)
    
    :param raw: raw VPixx data including eye blink channels. 
    :param pick: Label of eye blink channel in VPixx data. Should be same eye as used in MEG.
    :param threshold: Threshold to detect blinks
    :return: Binary vector indicating blink presence
    '''
    eye_blink = raw.get_data(picks=[pick])
    eye_blink = np.any(eye_blink > threshold, axis=0)
    return eye_blink

#%%
def binarize_binvector(x_bin):
    '''
    Clean a binary vector using morphological opening and closing.

    This function removes isolated single-sample spikes and fills small gaps
    in a binary (0/1) time series by applying a binary opening followed by
    a binary closing with a 1D structuring element of length 3.

    Parameters
    ----------
    x_bin : array_like
        Input binary vector (e.g., blink or event indicator). Values are
        interpreted as binary after conversion to float.

    Returns
    -------
    x : ndarray
        Cleaned binary vector of the same shape as `x_bin`, with spurious
        single-sample events removed and short gaps closed.

    Notes
    -----
    This function is typically used to preprocess binary event channels
    (e.g., eye blinks) prior to onset detection or cross-modal alignment.

    The operation performed is:
    binary_closing(binary_opening(x_bin, structure=[1, 1, 1])).
    '''
    x = x_bin.astype(float)
    x = binary_closing(binary_opening(x, structure=np.ones(3)))
    return x

#%%
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

#%%
#%%

def calcoffset_coarse(meg_z, eye_z, sfreq =1000, max_lag_sec=45):
    '''
    Compute first coarse offset between MEG data and VPIXX data.
    
    :param meg_z: z transformed meg eye blink data
    :param eye_z: z transformed vpixx eye blink data
    :param sfreq: common sampling rate (HAS TO BE ASSURED BEFORE)
    :param max_lag_sec: maximum lag to consider (in seconds)
    '''
    max_lag = int(max_lag_sec * sfreq)

    corr = correlate(meg_z, eye_z, mode="full")
    lags = np.arange(-len(eye_z) + 1, len(meg_z))

    mask = np.abs(lags) <= max_lag
    best_lag = lags[mask][np.argmax(corr[mask])]

    offset_sec = best_lag / sfreq
    print(f"Coarse offset: {offset_sec:.3f} s")
    return offset_sec, best_lag

#%%
def blink_onsets_from_binary(sig):
    diff = np.diff(sig.astype(int), prepend=0)
    return np.where(diff == 1)[0]

#%%
import numpy as np
from scipy.spatial.distance import cdist

def finematchingblinks(meg_onsets, eye_onsets, best_lag, sfreq=1000, tolerace=0.2):
    '''
    Fine-match blink onsets between MEG and eye tracking and estimate offset/drift.

    Parameters
    ----------
    meg_onsets : array_like
        Blink onset sample indices in MEG sample space.
    eye_onsets : array_like
        Blink onset sample indices in the same sample space as `meg_onsets`.
    best_lag : int
        Coarse lag in samples to shift eye onsets toward MEG onsets.
    sfreq : float
        Sampling frequency in Hz.
    tolerace : float
        Maximum allowed mismatch in seconds for a valid blink pair.

    Returns
    -------
    matchres : dict
        Dictionary with keys:
        - 'slope', 'intercept', 'r2'
        - 'refined_offset_sec'
        - 't_eye', 't_meg' (matched onset times in seconds)
        - 'n_matched' (number of matched blinks)

    Notes
    -----
    The algorithm proceeds as follows:

    1. Shift eye-tracking blink onsets by the coarse lag.
    2. Match each MEG blink to the nearest shifted eye-tracking blink.
    3. Reject pairs whose temporal discrepancy exceeds the specified tolerance.
    4. Estimate a refined temporal offset using the median residual.
    5. Fit a linear model to estimate relative clock drift and intercept.
    6. Compute R² to quantify alignment quality.

    A drift factor close to 1.0 and an R² close to 1.0 indicate excellent temporal
    alignment and stable clocks across recording systems.

    This function is intended for high-precision MEG–eye-tracking alignment in
    the absence of hardware triggers, using blink events as shared physiological
    landmarks.
    '''

    meg_onsets = np.asarray(meg_onsets, dtype=int)
    eye_onsets = np.asarray(eye_onsets, dtype=int)

    eye_onsets_shifted = eye_onsets + int(best_lag)

    # nearest-neighbor matching
    D = cdist(meg_onsets[:, None], eye_onsets_shifted[:, None], metric="euclidean")
    pairs = np.argmin(D, axis=1)

    residuals = meg_onsets - eye_onsets_shifted[pairs]
    good = np.abs(residuals) < int(tolerace * sfreq)

    matched_meg = meg_onsets[good]
    matched_eye = eye_onsets[pairs[good]]  # unshifted eye indices, for reporting

    matchres = {}

    # Guard: need at least 2 points for a drift fit, and at least 1 for offset median
    if matched_meg.size == 0:
        matchres.update(
            slope=np.nan,
            intercept=np.nan,
            r2=np.nan,
            refined_offset_sec=np.nan,
            t_eye=np.array([]),
            t_meg=np.array([]),
            n_matched=0,
        )
        return matchres

    refined_offset_sec = (best_lag + np.median(residuals[good])) / sfreq

    matchres["refined_offset_sec"] = float(refined_offset_sec)
    matchres["t_eye"] = matched_eye / sfreq
    matchres["t_meg"] = matched_meg / sfreq
    matchres["n_matched"] = int(matched_meg.size)

    if matched_meg.size < 2:
        matchres.update(slope=np.nan, intercept=np.nan, r2=np.nan)
        return matchres

    # drift fit: meg_sample ≈ slope * eye_sample + intercept
    slope, intercept = np.polyfit(matched_eye, matched_meg, 1)

    matched_meg_hat = slope * matched_eye + intercept
    ss_res = np.sum((matched_meg - matched_meg_hat) ** 2)
    ss_tot = np.sum((matched_meg - np.mean(matched_meg)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    matchres["slope"] = float(slope)
    matchres["intercept"] = float(intercept)
    matchres["r2"] = float(r2)

    return matchres

#%%

def _ensure_sfreq(raw, target_sfreq, label="", verbose=True):
    current_sfreq = raw.info["sfreq"]
    if not np.isclose(current_sfreq, target_sfreq):
        if verbose:
            print(
                f"{label} sfreq={current_sfreq:.2f} Hz -> "
                f"resampling to {target_sfreq:.2f} Hz"
            )
        raw = raw.copy().resample(
            target_sfreq,
            npad="auto",
            verbose="error",
        )
    return raw

def equalize_event_density_timebins(
    t_eye,
    t_meg,
    bin_width_sec=20.0,
    criterion="center",
    min_events_per_bin=1,
):
    """
    Downsample events so that each fixed-duration time bin
    contributes approximately the same number of events.

    Dense bins are capped; sparse bins are kept as-is.
    """
    t_eye = np.asarray(t_eye)
    t_meg = np.asarray(t_meg)

    t_min = t_eye.min()
    t_max = t_eye.max()

    edges = np.arange(t_min, t_max + bin_width_sec, bin_width_sec)
    keep_idx = []

    # collect indices per bin
    bins = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        idx = np.where((t_eye >= b0) & (t_eye < b1))[0]
        if len(idx) >= min_events_per_bin:
            bins.append(idx)

    if not bins:
        raise RuntimeError("No bins contain enough events")

    # cap = minimum non-empty bin size
    cap = min(len(idx) for idx in bins)

    for idx in bins:
        if len(idx) <= cap:
            keep_idx.extend(idx)
        else:
            if criterion == "center":
                center = idx[len(idx) // 2]
                keep_idx.append(center)
            elif criterion == "jitter":
                jitter = np.abs(t_eye[idx] - t_meg[idx])
                keep_idx.append(idx[np.argmin(jitter)])
            elif criterion == "random":
                keep_idx.extend(
                    np.random.choice(idx, cap, replace=False)
                )
            else:
                raise ValueError("Unknown criterion")

    keep_idx = np.sort(np.asarray(keep_idx))
    return t_eye[keep_idx], t_meg[keep_idx]




def wrapper_align_by_blinks(
    rawMEG,
    rawVPixx,
    meg_picks=("MISC010", "MISC011"),
    meg_threshold_percentile=99,
    vpixx_pick="Left Eye Blink",
    vpixx_threshold=0.5,
    sfreq=1000,
    max_lag_sec=45,
    tolerance_sec=0.2,
    uniform=False,
    secbin=20,
    verbose=True,
):

    # --- Ensure consistent sampling frequency ---
    rawMEG = _ensure_sfreq(rawMEG, sfreq, label="MEG", verbose=verbose)
    rawVPixx = _ensure_sfreq(rawVPixx, sfreq, label="VPixx", verbose=verbose)

    if verbose:
        print(f"Using common sfreq = {sfreq} Hz")

    # --- Blink extraction ---
    meg_blink_bin = blinkfromMEG(
        rawMEG, picks=list(meg_picks), threshold_percentile=meg_threshold_percentile
    )
    eye_blink = blinkfromVPixx(
        rawVPixx, pick=vpixx_pick, threshold=vpixx_threshold
    )

    meg = binarize_binvector(meg_blink_bin)
    eye = binarize_binvector(eye_blink)

    meg_z = zscore(meg.astype(float))
    eye_z = zscore(eye.astype(float))

    offset_sec, best_lag = calcoffset_coarse(
        meg_z, eye_z, sfreq=sfreq, max_lag_sec=max_lag_sec
    )

    meg_onsets = blink_onsets_from_binary(meg)
    eye_onsets = blink_onsets_from_binary(eye)

    matchres = finematchingblinks(
        meg_onsets,
        eye_onsets,
        best_lag,
        sfreq=sfreq,
        tolerace=tolerance_sec,
    )

    if uniform:
        print("Using density-equalized blink matches for realignment.")
        t_eye, t_meg = equalize_event_density_timebins(
            matchres["t_eye"],
            matchres["t_meg"],
            bin_width_sec=secbin,
            criterion="jitter",
        )
    else:
        print("Using all blink matches for realignment.")
        t_eye = matchres["t_eye"]
        t_meg = matchres["t_meg"]

    
    mne.preprocessing.realign_raw(
        rawVPixx,
        rawMEG,
        t_eye,
        t_meg,
        verbose="error",
    )
    
    '''
    mne.preprocessing.realign_raw(
        rawMEG,
        rawVPixx,
        t_meg,
        t_eye,
        verbose="error",
    )
    '''
    rawAll = rawMEG.copy()
    rawAll.add_channels([rawVPixx], force_update_info=True)

    return rawAll, matchres, offset_sec, best_lag
