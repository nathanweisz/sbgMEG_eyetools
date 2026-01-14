#%%
import mne
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, blink_stats_from_annotations, call_blink_annotations
from eyetools.alignETMEGbyblinks import wrapper_align_by_blinks
from eyetools.shiftsignals import align_raw_by_continuous_lag

#%% Uses same info for each subject
#TO DO: write reader for VPixx calibration file
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

rawVPixx = make_eye_mne(dataVpixx, srate)
# conversion not applied to Raw x/y
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

#%%
rawVPixx.plot(picks=['Left Eye y', 'Right Eye y'])
plt.close("all")

#%% Define channels in which to consider blinks
BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(rawVPixx, BLINK_MAP)
rawVPixx.set_annotations(annotations)

# %%
mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)
# Downsample to MEG sampling rate
#rawVPixx.resample(1000);

# %%
rawVPixx.plot(picks=['Left Eye x', 'Left Eye y',
                           'Right Eye x', 'Right Eye y'])
plt.close("all")

# %%
stats = blink_stats_from_annotations(rawVPixx_clean.annotations)

plt.hist(stats['left']['durations'], bins=20)
plt.hist(stats['right']['durations'], bins=20)

plt.hist(stats['left']['ibi'], bins=20)
plt.hist(stats['right']['ibi'], bins=20)


#%% LOAD MEG eye channels
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%% pupil 'MISC012']
rawMEG.plot(picks=['MISC010', 'MISC011', 'EOG001', 'EOG002'])
plt.close("all")
#--> BLINKS CLEAR IN EOG ... MISC A LOT OF SHIT

#%%
#ADDING UNIFORM AND BIN SIZE SEEMS TO STABILIZE DELAYS (i.e. BEING MORE CONSTANT) WHEN DATA LOSSES STRONGLY UNEVENLY SPACED
rawAll2, matchres2, offset_sec2, best_lag2 = wrapper_align_by_blinks(rawMEG.copy(), rawVPixx.copy(), 
                                                                     meg_threshold_percentile=99.5,
                                                                     uniform=True, secbin=5)

#%%
#see above ... i still have a smallish delauf of ~76ms
rawAligned, lag_info = align_raw_by_continuous_lag(rawAll2)
                           
#%%
rawAligned.filter(None, 30, fir_design='firwin')

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
