# %%
eye_data, srate = readvpixxmat('/home/aetzler/ocular-music-tracking/data_synced/Eye_data/19910703eigl/music_01.mat')
dig = eye_data[:, 9]     # adjust index if different
if (dig > 256).sum() > 1:
    eye_data[:, 9] /= 256

eye_data[eye_data == 9999.0] = np.nan
eye_mne = make_eye_mne(eye_data, srate)
cals = vpixx_templatecalibration(size = (0.56, 0.29))
raw_eye = mne.preprocessing.eyetracking.convert_units(eye_mne, calibration=cals, to="radians")      # screen size wrong? maybe should be 0.61, 0.34?
BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(raw_eye, BLINK_MAP)
raw_eye.set_annotations(annotations)
raw_eye = mne.preprocessing.eyetracking.interpolate_blinks(
raw_eye, buffer=(0.05, 0.2), interpolate_gaze=True
)
# no need to resample (realign_raw does it internally)
raw_eye.set_channel_types({'Digital Output': 'stim'})

# make sure triggers from eye tracking and meg are the same in length
eye_events = mne.find_events(raw_eye, stim_channel = 'Digital Output', initial_event = True)
# %%
meg_raw = mne.io.read_raw_fif('/home/aetzler/ocular-music-tracking/data_synced/sinuhe/240503/19910703eigl_block01.fif', preload = True)
meg_events = mne.find_events(meg_raw, stim_channel = 'STI101', initial_event = True)
meg_events = meg_events[meg_events[:, 2] < 4096] # filtered out answer triggers cause they werent in the eye data

#%%
meg_samples = meg_events[:, 0] - meg_raw.first_samp
t_meg = meg_samples / meg_raw.info["sfreq"]
#%%
eye_samples = eye_events[:, 0]
t_eye = eye_samples / raw_eye.info['sfreq']

# %%
mne.preprocessing.realign_raw(
    meg_raw,
    raw_eye,

    t_raw = t_meg,
    t_other = t_eye,
    verbose="error",
)

#%%
meg_raw.add_channels([raw_eye], force_update_info=True)

# %%
meg_raw.plot(picks = ['EOG001', 'EOG002', 'MISC011', 'MISC012', 'Left Eye x', 'Left Eye y'])
# %%