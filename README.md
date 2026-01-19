# sbgMEG-eyetools

A collection of tools to read TrackPixx data and align with the Salzburg MEG data.


This is how you get your data into an MNE raw object:
```python
from eyetools.annotateblinks import vpixx_default_blinkmap, blink_stats_from_annotations, call_blink_annotations
import eyetools.alignETMEGbyblinks as alignETMEGbyblinks

#%% Uses same info for each subject
#TO DO: write reader for VPixx calibration file
dataVpixx, srate = readvpixxmat('data/resting_vpixx.mat')

rawVPixx = make_eye_mne(dataVpixx, srate)
# conversion not applied to Raw x/y
cals = vpixx_templatecalibration()
rawVPixx = mne.preprocessing.eyetracking.convert_units(rawVPixx, calibration=cals, to="radians")

#Blink in VPIXX terms means “data loss“
BLINK_MAP = vpixx_default_blinkmap()  
annotations = call_blink_annotations(rawVPixx, BLINK_MAP)
rawVPixx.set_annotations(annotations)

mne.preprocessing.eyetracking.interpolate_blinks(
    rawVPixx, buffer=(0.02, 0.1), interpolate_gaze=True
)

rawVPixx.plot(picks=['Left Eye y', 'Right Eye y'])
```

Check out the folder [examples](examples/) for more specific use cases. This folder contains jupyter notebooks, walking you through some analysis steps.
They are however incomplete. So you may also find the information you need in the [tests](tests/) folder.

