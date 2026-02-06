#%%
import mne
import numpy as np
from matplotlib import pyplot as plt
from eyetools.annotateblinks import add_blinkvec2raw 

#%%
raw = mne.io.read_raw_fif('data/20020127evab_resting-raw.fif', preload=True)

#%%
raw, blinks_df = add_blinkvec2raw(raw,thresh=50)

# %%
raw.plot(
    picks=[
        "EOG001",
        "BLINK",
    ],
    scalings={
        "misc": 0.1,
        "eog": 400e-6,
        "eyegaze": 0.05,
        "pupil": 5
    },
)
# %%
