# sbgMEG-eyetools

A collection of tools to read TrackPixx data and align with the Salzburg MEG data.

```python
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
```
