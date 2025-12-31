#%%
import mne
from mne.viz.eyetracking import plot_gaze
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import blinks_to_annotations, vpixx_default_blinkmap, blink_stats_from_annotations
from scipy.signal import hilbert

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
rawVPixx.plot(picks=['Left Eye x', 'Left Eye y'])
plt.close("all")

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

# %%


#%% LOAD MEG eye channels
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%% pupil 'MISC012']
rawMEG.plot(picks=['MISC010', 'MISC011', 'EOG001', 'EOG002'])
plt.close("all")
#--> BLINKS CLEAR IN EOG ... MISC A LOT OF SHIT

# %%
meg_blink = rawMEG.get_data(picks=['MISC010', 'MISC011'])
meg_blink = np.mean(meg_blink, axis=0)
meg_env = np.abs(hilbert(meg_blink))
meg_blink_bin = meg_env > np.percentile(meg_env, 99)

#%% WHICH EYE MEASURED IN MEG??
eye_blink = rawVPixx.get_data(picks=['Left Eye Blink'])
eye_blink = np.any(eye_blink > 0.5, axis=0)

# %%
meg = meg_blink_bin.astype(float)
eye = eye_blink.astype(float)

from scipy.ndimage import binary_opening, binary_closing

meg = binary_closing(binary_opening(meg, structure=np.ones(3)))
eye = binary_closing(binary_opening(eye, structure=np.ones(3)))

#%%
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

meg_z = zscore(meg.astype(float))
eye_z = zscore(eye.astype(float))

#%%
from scipy.signal import correlate

sfreq_meg = rawMEG.info['sfreq']

max_lag_sec = 30
max_lag = int(max_lag_sec * sfreq_meg)

corr = correlate(meg_z, eye_z, mode="full")
lags = np.arange(-len(eye_z) + 1, len(meg_z))

mask = np.abs(lags) <= max_lag
best_lag = lags[mask][np.argmax(corr[mask])]

offset_sec = best_lag / sfreq_meg
print(f"Coarse offset: {offset_sec:.3f} s")

# %%

def blink_onsets_from_binary(sig):
    diff = np.diff(sig.astype(int), prepend=0)
    return np.where(diff == 1)[0]

meg_onsets = blink_onsets_from_binary(meg)
eye_onsets = blink_onsets_from_binary(eye)

#%%
eye_onsets_shifted = eye_onsets + best_lag

#%%
from scipy.spatial.distance import cdist

D = cdist(
    meg_onsets[:, None],
    eye_onsets_shifted[:, None],
    metric="euclidean"
)

pairs = np.argmin(D, axis=1)
residuals = meg_onsets - eye_onsets_shifted[pairs]

#%%
good = np.abs(residuals) < int(0.2 * sfreq_meg)  # 200 ms tolerance
residuals = residuals[good]

#%%

refined_offset_sec = (
    best_lag + np.median(residuals)
) / sfreq_meg

print(f"Refined offset: {refined_offset_sec:.4f} s")

# %%
matched_meg = meg_onsets[good]
matched_eye = eye_onsets[pairs[good]]

coef = np.polyfit(matched_eye, matched_meg, 1)
slope, intercept = coef

print(f"Drift factor: {slope:.8f}")
print(f"Intercept (samples): {intercept:.1f}")

#%%

t_eye = matched_eye / sfreq_meg
t_meg = matched_meg / sfreq_meg

#%%

mne.preprocessing.realign_raw(
    rawVPixx,
    rawMEG,
    t_eye,
    t_meg,
    verbose="error",
)

#%%
rawAll = rawMEG.copy()
rawAll.add_channels([rawVPixx], force_update_info=True)

del rawMEG

#%%
rawAll.plot(
    picks=['MISC010', 'MISC011', 'EOG001', 'EOG002',
           'Left Eye x', 'Left Eye y'],
    scalings={
        'misc': .1,      # adjust to taste
        'eog': 400e-6,     # ~200 µV
        'eyegaze': 0.05,   # ~0.05 rad (or ~3 deg)
    }
)

plt.close("all")

# %%
