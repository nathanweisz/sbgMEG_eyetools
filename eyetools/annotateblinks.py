#%%
import numpy as np

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
