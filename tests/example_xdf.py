#%%imports
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne

#%%define stuff
data_path = Path('./data/example_eeg.xdf')
labrec, header = pyxdf.load_xdf(data_path)

#%% load data

stream_dict = dict()
for idx_labrec, stream in enumerate(labrec):
            which_stream = labrec[:][idx_labrec]['info']['name'][0]
            stream_dict[which_stream] = idx_labrec

eeg_data = labrec[stream_dict['EEG']]['time_series']
#convert from microVolts to Volts
eeg_data = eeg_data * 1e-6
eeg_time = labrec[stream_dict['EEG']]['time_stamps']
trigger = labrec[stream_dict['o_ptb_marker_stream']]['time_series']
trigger_time = labrec[stream_dict['o_ptb_marker_stream']]['time_stamps']
eye_data = labrec[stream_dict['pupil_capture']]['time_series']
eye_time = labrec[stream_dict['pupil_capture']]['time_stamps']

#%%

channels_desc = (
    labrec[stream_dict['pupil_capture']]
    ['info']['desc'][0]['channels'][0]['channel']
)

ch_names = [ch['label'][0] for ch in channels_desc]

#%%
threshold_confidence = 0.8
low_conf_flag = (eye_data[:, 0] <= threshold_confidence).astype(int)
eye_data = np.column_stack([eye_data, low_conf_flag])
ch_names.append("low_confidence")

#%%
mask = eye_data[:, 0] > threshold_confidence

eye_data_filt = eye_data[mask, :]
eye_time_filt = eye_time[mask]


#%%
def lsl2mneraw(
    eeg_data,
    eeg_time,
    eye_data,
    eye_time,
    eye_ch_names,
    *,
    trigger=None,
    trigger_time=None,
    eeg_ch_names=None,
    eeg_ch_types=None,
    sort_times=True,
    fill_value=np.nan,
):
    """
    Merge EEG, eye-tracking, and triggers into a single MNE Raw object.
    """

    # -------------------------
    # helpers
    # -------------------------
    def resample_eye_to_eeg_time(eeg_time, eye_time, eye_data):
        if sort_times:
            order = np.argsort(eye_time)
            eye_time = eye_time[order]
            eye_data = eye_data[order]

        out = np.full((len(eeg_time), eye_data.shape[1]), fill_value)
        for ch in range(eye_data.shape[1]):
            sig = eye_data[:, ch]
            valid = np.isfinite(sig)
            if valid.sum() >= 2:
                out[:, ch] = np.interp(
                    eeg_time,
                    eye_time[valid],
                    sig[valid],
                    left=fill_value,
                    right=fill_value,
                )
        return out

    def infer_eye_ch_types(names):
        types = []
        for ch in names:
            if ch.startswith("diameter"):
                types.append("pupil")
            elif ch.startswith(("norm_pos", "gaze_point", "gaze_normal")):
                types.append("eyegaze")
            else:
                types.append("misc")
        return types

    # -------------------------
    # EEG Raw
    # -------------------------
    eeg_time = np.asarray(eeg_time)
    eeg_data = np.asarray(eeg_data)

    sfreq = 1.0 / np.median(np.diff(np.sort(eeg_time)))

    if eeg_ch_names is None:
        eeg_ch_names = [f"EEG{idx:03d}" for idx in range(eeg_data.shape[1])]
    if eeg_ch_types is None:
        eeg_ch_types = ["eeg"] * eeg_data.shape[1]

    info_eeg = mne.create_info(eeg_ch_names, sfreq, eeg_ch_types)
    raw = mne.io.RawArray(eeg_data.T, info_eeg, verbose=False)

    # -------------------------
    # Eye Raw
    # -------------------------
    eye_on_eeg = resample_eye_to_eeg_time(eeg_time, eye_time, eye_data)
    eye_ch_types = infer_eye_ch_types(eye_ch_names)

    info_eye = mne.create_info(eye_ch_names, sfreq, eye_ch_types)
    raw_eye = mne.io.RawArray(eye_on_eeg.T, info_eye, verbose=False)

    raw.add_channels([raw_eye], force_update_info=True)

    # -------------------------
    # Triggers → Annotations
    # -------------------------
    if trigger is not None and trigger_time is not None:
        trigger = np.asarray(trigger).squeeze()
        trigger_time = np.asarray(trigger_time)

        # EEG time zero as reference
        t0 = eeg_time.min()
        onsets = trigger_time - t0
        durations = np.zeros_like(onsets)
        descriptions = [str(int(v)) for v in trigger]

        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)

    return raw

#%%

raw = lsl2mneraw(
    eeg_data,
    eeg_time,
    eye_data_filt,
    eye_time_filt,
    ch_names,
    trigger=trigger,
    trigger_time=trigger_time,)

raw.filter(.5, 25, picks='eeg')
raw.filter(None, 6, picks='pupil')
raw.filter(None, 25, picks='eyegaze')

#%%

raw.plot(picks=['gaze_normal0_x', 'gaze_normal0_y', 
                'gaze_normal1_x', 'gaze_normal1_y',
                'diameter0_2d', 'diameter1_2d'], scalings=dict(eyegaze=0.5, pupil=15))
# %%
raw.plot(picks=['low_confidence'], scalings=dict(misc=1))


# %%
raw.plot(picks='eeg', scalings=dict(eeg = 50e-6))
# %%
