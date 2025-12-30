


#%%

#%% LOAD MEG eye channels
rawMEG = mne.io.read_raw_fif('data/resting_meg_trans_sss.fif', preload=True)

#%% pupil 'MISC012']
rawMEG.plot(picks=['MISC010', 'MISC011'])

# %%
meg_blink = rawMEG.get_data(picks=['MISC010', 'MISC011'])
meg_blink = np.mean(meg_blink, axis=0)
meg_env = np.abs(hilbert(meg_blink))
meg_blink_bin = meg_env > np.percentile(meg_env, 99)

#%% WHICH EYE MEASURED IN MEG??
eye_blink = rawVPixx.get_data(picks=['Left Eye Blink'])
eye_blink = np.any(eye_blink > 0.5, axis=0)
