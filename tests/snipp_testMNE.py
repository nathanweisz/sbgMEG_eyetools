#%%
import mne
from mne.datasets.eyelink import data_path
from mne.viz.eyetracking import plot_gaze
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt

et_fpath = data_path() / "eeg-et" / "sub-01_task-plr_eyetrack.asc"
raw_eyelink = mne.io.read_raw_eyelink(et_fpath, create_annotations=["blinks"])

#%%
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

# %%
rawVPixx = make_eye_mne(dataVpixx, srate)

#%%
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")
#%%
#rawVPixx.plot()
rawVPixx.plot(picks=['Left Eye Blink', 'Right Eye Blink'])

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
    starts = np.where(diff == 1)[1]
    ends = np.where(diff == -1)[1]

    if signal.any() == 1:
        ends = np.append(ends, len(signal))

    return list(zip(starts, ends))

#%%

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

#%%
def blinks_to_annotations(raw, blink_channel, affected_channels):
    sfreq = raw.info["sfreq"]
    blink_data = raw.get_data(picks=[blink_channel])[0]

    segments = find_binary_segments(blink_data)

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
lblinks = rawVPixx.get_data(picks=['Left Eye Blink'])  
test = find_blink_samples(lblinks)
# %%
