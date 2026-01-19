#%%
import mne
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, blink_stats_from_annotations, call_blink_annotations
from eyetools.alignETMEGbyblinks import wrapper_align_by_blinks, find_meg_blink_thresholds
from eyetools.shiftsignals import align_raw_by_continuous_lag
mne.viz.set_browser_backend("qt")

#%% Uses same info for each subject
#TO DO: write reader for VPixx calibration file
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

rawVPixx = make_eye_mne(dataVpixx, srate)

cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

#%% Define channels in which to consider blinks
BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(rawVPixx, BLINK_MAP)
rawVPixx.set_annotations(annotations)

mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)

#%% LOAD MEG eye channels
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%%
thresh, offsets, correlations = find_meg_blink_thresholds(rawMEG, rawVPixx)

# %%

rawAll2, matchres2, offset_sec2, best_lag2 = wrapper_align_by_blinks(rawMEG.copy(), rawVPixx.copy(), 
                                                                     meg_threshold_percentile=thresh,
                                                                     uniform=True, secbin=5)

#%%
rawAligned, lag_info = align_raw_by_continuous_lag(rawAll2)


#%%

rawAligned.plot(
    picks=[
        "MISC010", "MISC011",
        "Left Eye x", "Left Eye y", "Left Eye Pupil Diameter",
        "Right Eye x", "Right Eye y", "Right Eye Pupil Diameter",
    ],
    scalings={
        "misc": 0.1,
        "eog": 400e-6,
        "eyegaze": 0.05,
        "pupil": 5
    },
)

plt.close("all")
                                               

# %%
