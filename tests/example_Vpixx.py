#%%
import mne
from mne.viz.eyetracking import plot_gaze
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import blinks_to_annotations, vpixx_default_blinkmap
from scipy.signal import hilbert

#et_fpath = data_path() / "eeg-et" / "sub-01_task-plr_eyetrack.asc"
#raw_eyelink = mne.io.read_raw_eyelink(et_fpath, create_annotations=["blinks"])

#%% Uses same info for each subject
#TO DO: write reader for VPixx calibration file
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

# %% Convert to mne Raw object
rawVPixx = make_eye_mne(dataVpixx, srate)

#%% conversion not applied to Raw x/y
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

#%%
rawVPixx.plot(picks=['Left Eye x', 'Left Eye y'])

#%% Define channels in which to consider blinks
BLINK_MAP = vpixx_default_blinkmap()

# %%
all_annotations = []

for blink_ch, affected in BLINK_MAP.items():
    print(f"Processing {blink_ch} affecting {affected}")
    anns = blinks_to_annotations(rawVPixx, blink_ch, affected)
    all_annotations.extend(anns)

# Combine into a single Annotations object (MNE annotions object has neat .add feature)
annotations = sum(all_annotations[1:], all_annotations[0])

rawVPixx.set_annotations(annotations)

# %%
rawVPixx_clean = mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)
# %% Downsample to MEG sampling rate
rawVPixx_clean.resample(1000);

# %%
rawVPixx_clean.plot(picks=['Left Eye x', 'Left Eye y'])

# %%
