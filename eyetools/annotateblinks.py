#%%
import numpy as np
import mne
import pandas as pd
import neurokit2 as nk

# %% find on and offsets of blinks
def find_blink_samples(signal):
    """
    Find contiguous segments where signal == 1.

    Returns
    -------
    segments : list of (start_idx, end_idx)
    """
    signal = np.asarray(signal, dtype=int)

    diff = np.diff(signal, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if signal.any() == 1:
        ends = np.append(ends, len(signal))

    return list(zip(starts, ends))

#%%
def blinks_to_annotations(raw, blink_channel, affected_channels):
    sfreq = raw.info["sfreq"]
    blink_data = raw.get_data(picks=[blink_channel])[0]

    segments = find_blink_samples(blink_data)

    annotations = []

    for start, end in segments:
        onset = start / sfreq
        duration = (end - start) / sfreq

        ann = mne.Annotations(
            onset=[onset],
            duration=[duration],
            description=["BAD_blink"],
            ch_names=[affected_channels],
        )
        annotations.append(ann)

    return annotations

#%%
def call_blink_annotations(raw, BLINK_MAP):
    all_annotations = []

    for blink_ch, affected in BLINK_MAP.items():
        print(f"Processing {blink_ch} affecting {affected}")
        anns = blinks_to_annotations(raw, blink_ch, affected)
        all_annotations.extend(anns)

    # Combine into a single Annotations object (MNE annotions object has neat .add feature)
    annotations = sum(all_annotations[1:], all_annotations[0])
    return annotations

# %%
def vpixx_default_blinkmap():
    BLINK_MAP = {
        "Left Eye Blink": (
            "Left Eye x",
            "Left Eye y",
            "Left Eye Raw x",
            "Left Eye Raw y",
            "Left Eye Pupil Diameter",
        ),
        "Right Eye Blink": (
            "Right Eye x",
            "Right Eye y",
            "Right Eye Raw x",
            "Right Eye Raw y",
            "Right Eye Pupil Diameter",
        ),
    }
    return BLINK_MAP

# %% This is pure ChatGPT: unchecked.

def _eye_from_ch_names(ch_names):
    ch_names = [ch.lower() for ch in ch_names]
    left = any("left" in ch for ch in ch_names)
    right = any("right" in ch for ch in ch_names)
    return left, right

def blink_stats_from_annotations(annotations):
    """
    Extract blink statistics from mne.Annotations.

    Returns
    -------
    stats : dict
        Nested dict with keys 'left' and 'right', each containing:
        - n_blinks
        - durations (np.ndarray)
        - ibi (np.ndarray)
    """
    onsets = {"left": [], "right": []}
    durations = {"left": [], "right": []}

    for ann in annotations:
        if ann["description"] not in {"BAD_blink", "blink"}:
            continue

        left, right = _eye_from_ch_names(ann["ch_names"])

        if left:
            onsets["left"].append(float(ann["onset"]))
            durations["left"].append(float(ann["duration"]))

        if right:
            onsets["right"].append(float(ann["onset"]))
            durations["right"].append(float(ann["duration"]))

    stats = {}

    for eye in ("left", "right"):
        o = np.array(onsets[eye])
        d = np.array(durations[eye])

        if len(o) > 1:
            order = np.argsort(o)
            o = o[order]
            d = d[order]
            ibi = np.diff(o)
        else:
            ibi = np.array([])

        stats[eye] = {
            "n_blinks": len(o),
            "durations": d,
            "ibi": ibi,
        }

    return stats

#%%
#%%
def get_blinks_eog_infos(
    eog,
    sampling_rate: float = 1000,
    threshold_percentile: float = 75,
    window_samples: int | None = None,
):
    """Extract blink onsets, offsets, and durations from EOG signal."""
    
    if window_samples is None:
        window_samples = int(sampling_rate // 2)
    
    eog = np.abs(eog)
    eog_signals, info = nk.eog_process(eog, sampling_rate=sampling_rate)
    blink_peaks = np.where(eog_signals['EOG_Blinks'] == 1)[0]
    
    eog_clean_abs = np.abs(eog_signals['EOG_Clean'].values)
    threshold = np.percentile(eog_clean_abs, threshold_percentile)
    
    blink_onsets = []
    blink_offsets = []
    
    for peak in blink_peaks:
        onset = peak
        for i in range(peak, max(0, peak - window_samples), -1):
            if eog_clean_abs[i] < threshold:
                onset = i
                break
        
        offset = peak
        for i in range(peak, min(len(eog_signals), peak + window_samples)):
            if eog_clean_abs[i] < threshold:
                offset = i
                break
        
        blink_onsets.append(onset)
        blink_offsets.append(offset)
    
    return pd.DataFrame({
        'peak_samples': blink_peaks,
        'onset_samples': blink_onsets,
        'offset_samples': blink_offsets,
        'onset_sec': np.array(blink_onsets) / sampling_rate,
        'offset_sec': np.array(blink_offsets) / sampling_rate,
        'duration_sec': (np.array(blink_offsets) - np.array(blink_onsets)) / sampling_rate
    })

#%%
def add_blinkvec2raw(
    raw,
    hp_freq=0.1,
    lp_freq=10,
    eoglab=None,
    thresh=75,
):
    """
    Detect eye blinks from an EOG channel and add them to an MNE Raw object
    as both a binary blink channel and blink annotations.

    Blinks are detected from a band-pass filtered EOG signal using a
    percentile-based amplitude threshold. Detected blink intervals are
    encoded as a binary time series (0 = no blink, 1 = blink) and appended
    to the Raw object as a new channel named ``'BLINK'``. In addition,
    blink onsets and durations are added as MNE Annotations with the
    description ``'blink'``.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw M/EEG recording containing at least one EOG channel.
        The object is modified in-place by adding a new channel and
        annotations.

    hp_freq : float, optional
        High-pass filter cutoff frequency (Hz). Defaults to 0.1 Hz.

    lp_freq : float, optional
        Low-pass filter cutoff frequency (Hz). Defaults to 10 Hz.

    eoglab : list of str | None, optional
        Channel names used for blink detection. If None, defaults to
        ``['EOG001']``.

    thresh : float, optional
        Percentile (0–100) used as the amplitude threshold for blink
        detection. Defaults to 75.

    Returns
    -------
    raw : mne.io.Raw
        The modified Raw object with an added ``'BLINK'`` channel and
        blink annotations.

    blinks_df : pandas.DataFrame
        DataFrame containing blink onset and duration information.
    """
    if eoglab is None:
        eoglab = ['EOG001']

    sfreq = raw.info["sfreq"]
    n_times = raw.n_times

    # --- Filter EOG (on a copy, no side effects)
    eog = (
        raw.copy()
        .filter(hp_freq, lp_freq, picks=eoglab)
        .get_data(picks=eoglab)
    )

    if eog.shape[0] > 1:
        eog = eog.mean(axis=0)  # explicit channel reduction
    else:
        eog = eog[0]

    # --- Blink detection
    blinks_df = get_blinks_eog_infos(
        eog,
        sampling_rate=sfreq,
        threshold_percentile=thresh,
    )

    # --- Build blink vector
    blink_vec = np.zeros(n_times, dtype=np.int8)

    onsets = np.clip(
        blinks_df["onset_samples"].to_numpy(int), 0, n_times
    )
    offsets = np.clip(
        blinks_df["offset_samples"].to_numpy(int), 0, n_times
    )

    for on, off in zip(onsets, offsets):
        blink_vec[on:off] = 1

    # --- Add blink channel
    info = mne.create_info(
        ch_names=["BLINK"],
        sfreq=sfreq,
        ch_types=["misc"],
    )

    blink_raw = mne.io.RawArray(blink_vec[np.newaxis, :], info)
    raw.add_channels([blink_raw], force_update_info=True)

    # --- Add annotations (append, do not overwrite)
    blink_ann = mne.Annotations(
        onset=blinks_df["onset_sec"],
        duration=blinks_df["duration_sec"],
        description=["blink"] * len(blinks_df),
    )

    raw.set_annotations(raw.annotations + blink_ann)

    return raw, blinks_df
