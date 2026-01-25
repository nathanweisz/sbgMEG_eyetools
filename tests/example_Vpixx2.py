#%% Try using tool to better define missing data periods in MICS channels

import mne
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, call_blink_annotations
from eyetools.alignETMEGbyblinks import wrapper_align_by_dataloss
from eyetools.shiftsignals import align_raw_by_continuous_lag
#%%

dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

rawVPixx = make_eye_mne(dataVpixx, srate)
# conversion not applied to Raw x/y
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(rawVPixx, BLINK_MAP)
rawVPixx.set_annotations(annotations)

mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%%
rawAll, matchres, offset_sec, best_lag = wrapper_align_by_dataloss(rawMEG.copy(), rawVPixx.copy())                                                               
rawAligned, lag_info = align_raw_by_continuous_lag(rawAll)

# %%
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
