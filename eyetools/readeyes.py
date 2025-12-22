#%%
from pymatreader import read_mat
import mne
import numpy as np

#%%

def readvpixxmat(filepath):
    """Read VPixx .mat eye-tracking data file.

    Parameters
    ----------
    filepath : str
        Path to the .mat file.

    Returns
    -------
    data : numpy array
    """

    data = read_mat(filepath)['data']
    #srate = round(1/(data[1,0]-data[0,0]))
    srate = 1/(data[1,0]-data[0,0])
    data[:,0] = data[:,0]-data[0,0]
    data = data[:,1:] 

    return data, srate


#%%
def make_eye_mne(eye_data, srate):

    columns = ['Left Eye x', 'Left Eye y', 'Left Eye Pupil Diameter', 'Right Eye x', 
            'Right Eye y', 'Right Eye Pupil Diameter', 'Digital Input', 'Left Eye Blink', 'Right Eye Blink',
            'Digital Output', 'Left Eye Fixation', 'Right Eye Fixation', 'Left Eye Saccade', 'Right Eye Saccade',
            'Message code', 'Left Eye Raw x', 'Left Eye Raw y', 'Right Eye Raw x', 'Right Eye Raw y']
        
    eye_info = mne.create_info(
                ch_names=columns,
                sfreq=srate,
                #ch_types=ch_types
                )
    eye_data = eye_data.T  #transposing the data to have channels in rows and timepoints in columns

    eye_raw = mne.io.RawArray(                                                      #creating raw object for the eye tracking data, containing the data itself and the info
                    data=eye_data,
                    info=eye_info)
    
    #Set channels
    eye_raw = mne.preprocessing.eyetracking.set_channel_types_eyetrack(eye_raw,
                    mapping={'Left Eye x': ('eyegaze', 'px', 'left', 'x'),
                             'Left Eye Raw x': ('eyegaze', 'px', 'left', 'x'),
                              'Left Eye y': ('eyegaze', 'px', 'left', 'y'),
                              'Left Eye Raw y': ('eyegaze', 'px', 'left', 'y'),
                              'Right Eye x': ('eyegaze', 'px', 'right', 'x'),
                              'Right Eye Raw x': ('eyegaze', 'px', 'right', 'x'),
                              'Right Eye y': ('eyegaze', 'px', 'right', 'y'),
                              'Right Eye Raw y': ('eyegaze', 'px', 'right', 'y'),
                              'Left Eye Pupil Diameter': ('pupil', 'au', 'left'),
                              'Right Eye Pupil Diameter': ('pupil', 'au', 'right')})

    return eye_raw

#%%
def vpixx_templatecalibration(resolution=(1920, 1080),
                              size=(0.61, 0.61),
                              distance=0.82):
    """Create a template calibration dictionary for VPixx eye-tracking data.
    The only input that matters is screen resolution, size and distance.

    Returns
    -------
    cals : MNE Calibration object.
    
    """
    cals = mne.preprocessing.eyetracking.Calibration(onset = -10, 
                                                 model ='HV5', 
                                                 eye = 'right', 
                                                 avg_error = 0, 
                                                 max_error =0,
                                                 positions= np.array([0, 0],),
                                                 offsets = np.array([0, 0]),
                                                 gaze = np.array([[0, 0],]),
                                                 screen_resolution = resolution,
                                                 screen_size = size,
                                                 screen_distance = distance)
    return cals


