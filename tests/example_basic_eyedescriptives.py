#%% Try using tool to better define missing data periods in MICS channels

import mne
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, call_blink_annotations, add_blinkvec2raw
from eyetools.eye_descriptives import extract_eye_gazedata, ocular_activity_measures, detect_velocity_peaks, compute_peak_statistics

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

rawVPixx.filter(None, 250, n_jobs=1)

#%%
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%%
raweyes, gazedata = extract_eye_gazedata(rawVPixx, rawMEG)

#%%

f, axes = plt.subplots(ncols=2, figsize=(8, 4))
maxf=50
axes[0].plot(gazedata['irasa_out'].freqs[:maxf], gazedata['irasa_out'].periodic[0,:maxf])
axes[1].loglog(gazedata['irasa_out'].freqs[:maxf], gazedata['irasa_out'].aperiodic[0,:maxf  ])

# %%
fs=2000
gaze_results = ocular_activity_measures(raweyes[0,:], raweyes[1,:], sampling_rate=fs, savgol_window=51,)

# %%
peaks, threshold, above, segments = detect_velocity_peaks(
    gaze_results["velocity"],
    fs=fs,
    threshold_factor=6,
    min_duration_ms=2,
    merge_gap_ms=10,
)

t = np.arange(len(gaze_results["velocity"])) / fs
in1 = 30000
in2 = 40000

vel = gaze_results["velocity"]
peak_mask = (peaks >= in1) & (peaks < in2)
peaks_win = peaks[peak_mask]

plt.figure(figsize=(10, 4))
plt.plot(t[in1:in2], vel[in1:in2], label="velocity")
plt.plot(t[peaks_win], vel[peaks_win], "ro", label="peaks")
plt.axhline(threshold, color="k", linestyle="--", label="threshold")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#%%

peak_stats = compute_peak_statistics(
    peaks,
    velocity=gaze_results["velocity"],
    fs=fs,
)


# %%
