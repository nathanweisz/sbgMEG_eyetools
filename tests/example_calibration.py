#%%
from eyetools.calibrationchecker import analyze_vpixx_polyresponse

#%%
calpathL = './data/calibration/19800616mrgu/PolyResponse_L.jpg'
calpathR = './data/calibration/19800616mrgu/PolyResponse_R.jpg'

#%%
calL_res = analyze_vpixx_polyresponse(calpathL, plot=True, verbose=True)
#%%
calR_res = analyze_vpixx_polyresponse(calpathR, plot=True, verbose=True)

# %%
