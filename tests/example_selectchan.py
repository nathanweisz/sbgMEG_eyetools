#%%
import mne
from eyetools.readeyes import readvpixxmat, make_eye_mne, vpixx_templatecalibration
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import vpixx_default_blinkmap, blink_stats_from_annotations, call_blink_annotations, find_blink_samples, get_blinks_eog_infos
from eyetools.alignETMEGbyblinks import wrapper_align_by_blinks
from eyetools.shiftsignals import align_raw_by_continuous_lag


#%% Uses same info for each subject
#TO DO: write reader for VPixx calibration file
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

rawVPixx = make_eye_mne(dataVpixx, srate)
# conversion not applied to Raw x/y
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

# Define channels in which to consider blinks
BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(rawVPixx, BLINK_MAP)
rawVPixx.set_annotations(annotations)

#
mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.1, 0.2), interpolate_gaze=True
)
# Downsample to MEG sampling rate
rawVPixx.resample(1000);

#%%

rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%%
rawAll2, matchres2, offset_sec2, best_lag2 = wrapper_align_by_blinks(rawMEG.copy(), rawVPixx.copy(), 
                                                                     meg_threshold_percentile=99.5,
                                                                     uniform=True, secbin=2
                                                                     )
del rawMEG, rawVPixx

#%%
rawAligned, lag_info = align_raw_by_continuous_lag(rawAll2)

rawAligned.filter(None, 30, fir_design='firwin') # dunno if needed

#%% EEG001 is vEOG
rawAligned.plot(picks=['EOG001', 'EOG002'])

#%% USE MNE FUNCTION TO FIND EOG BLINKS
eog_events = mne.preprocessing.find_eog_events(rawAligned, 512, ch_name='EOG001')

#%% SETS EVENTS ON PEAKS
rawAligned.plot(
    picks=['MISC010', 'MISC011', 'EOG001', 'EOG002',
           'Left Eye x', 'Left Eye y',
           'Right Eye x', 'Right Eye y'],
    scalings={
        'misc': .1,      # adjust to taste
        'eog': 400e-6,     # ~200 µV
        'eyegaze': 0.05,   # ~0.05 rad (or ~3 deg)
    },
    events=eog_events
)

# %% USE NEUROKIT FOR MORE INFOS
eogdataraw = rawAligned.get_data(picks=['EOG001'])

blinks_df = get_blinks_eog_infos(
    eogdataraw.flatten(),
    sampling_rate=rawAligned.info["sfreq"]
)

ann = mne.Annotations(onset=blinks_df['onset_sec'], 
            duration=blinks_df['duration_sec'], 
            description=['blink']*len(blinks_df['onset_sec']))

rawAligned.set_annotations(ann)

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
