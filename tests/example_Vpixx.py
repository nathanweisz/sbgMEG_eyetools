#%%
import mne
from mne.viz.eyetracking import plot_gaze
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, blink_stats_from_annotations, call_blink_annotations
import eyetools.alignETMEGbyblinks as alignETMEGbyblinks

#et_fpath = data_path() / "eeg-et" / "sub-01_task-plr_eyetrack.asc"
#raw_eyelink = mne.io.read_raw_eyelink(et_fpath, create_annotations=["blinks"])

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
rawVPixx_clean = mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)
# Downsample to MEG sampling rate
rawVPixx_clean.resample(1000);

# %%
rawVPixx_clean.plot(picks=['Left Eye x', 'Left Eye y',
                           'Right Eye x', 'Right Eye y'])
plt.close("all")

# %%
stats = blink_stats_from_annotations(rawVPixx_clean.annotations)

# %% MUCH LESS BLINKS RIGHT??
plt.hist(stats['left']['durations'], bins=20)
plt.hist(stats['right']['durations'], bins=20)
# %%
plt.hist(stats['left']['ibi'], bins=20)
plt.hist(stats['right']['ibi'], bins=20)


#%% LOAD MEG eye channels
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%% pupil 'MISC012']
rawMEG.plot(picks=['MISC010', 'MISC011', 'EOG001', 'EOG002'])
plt.close("all")
#--> BLINKS CLEAR IN EOG ... MISC A LOT OF SHIT

#%%
meg_blink_bin = alignETMEGbyblinks.blinkfromMEG(rawMEG)
eye_blink = alignETMEGbyblinks.blinkfromVPixx(rawVPixx)

meg=alignETMEGbyblinks.binarize_binvector(meg_blink_bin)
eye=alignETMEGbyblinks.binarize_binvector(eye_blink)

meg_z = alignETMEGbyblinks.zscore(meg.astype(float))
eye_z = alignETMEGbyblinks.zscore(eye.astype(float))

#%% CHECK THAT SAMPLING RATES MATCH
sfreq_meg = rawMEG.info['sfreq']
sfreq_vpixx = rawVPixx_clean.info['sfreq']
print(f"MEG srate: {sfreq_meg}, VPixx srate: {sfreq_vpixx}")

#%%
offset_sec, best_lag = alignETMEGbyblinks.calcoffset_coarse(meg_z, eye_z)

# %%
meg_onsets = alignETMEGbyblinks.blink_onsets_from_binary(meg)
eye_onsets = alignETMEGbyblinks.blink_onsets_from_binary(eye)

#%%
matchres = alignETMEGbyblinks.finematchingblinks(meg_onsets, eye_onsets, best_lag)

#%%
mne.preprocessing.realign_raw(
    rawVPixx,
    rawMEG,
    matchres['t_eye'],
    matchres['t_meg'],
    verbose="error",
)

#%%
rawAll = rawMEG.copy()
rawAll.add_channels([rawVPixx], force_update_info=True)

del rawMEG

#%%
rawAll.plot(
    picks=['MISC010', 'MISC011', 'EOG001', 'EOG002',
           'Left Eye x', 'Left Eye y',
           'Right Eye x', 'Right Eye y'],
    scalings={
        'misc': .1,      # adjust to taste
        'eog': 400e-6,     # ~200 µV
        'eyegaze': 0.05,   # ~0.05 rad (or ~3 deg)
    }
)

plt.close("all")

# %%
