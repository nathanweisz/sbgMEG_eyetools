import numpy as np
import mne
from scipy.signal import correlate


def measure_lag(eye, meg, sfreq=1000, maxlag_sec=0.5):
    eye = (eye - eye.mean()) / (eye.std() + 1e-12)
    meg = (meg - meg.mean()) / (meg.std() + 1e-12)

    max_lag = int(maxlag_sec * sfreq)

    corr = correlate(eye, meg, mode="full")
    corr /= np.sqrt(np.sum(eye**2) * np.sum(meg**2))

    lags = np.arange(-len(meg) + 1, len(eye))
    mask = np.abs(lags) <= max_lag

    best_lag = lags[mask][np.argmax(corr[mask])]
    lag_sec = best_lag / sfreq
    strength = np.max(corr[mask])

    return best_lag, lag_sec, strength


def apply_lag_shift_multichan(eye_data, meg_data, lag):
    if lag > 0:
        eye_shift = eye_data[:, lag:]
        meg_shift = meg_data[:, : eye_shift.shape[1]]
    elif lag < 0:
        meg_shift = meg_data[:, -lag:]
        eye_shift = eye_data[:, : meg_shift.shape[1]]
    else:
        eye_shift = eye_data
        meg_shift = meg_data

    return eye_shift, meg_shift


def align_raw_by_continuous_lag(
    rawAll,
    left_eye_ref_picks=("Left Eye x", "Left Eye y"),
    right_eye_ref_picks=("Right Eye x", "Right Eye y"),
    meg_ref_picks={"x": "MISC011", "y": "MISC010"},
    left_eye_all_picks=None,
    right_eye_all_picks=None,
    which_eye="left",        # "left", "right", or "best"
    sfreq=None,
    maxlag_sec=0.5,
    lag_strategy="best",     # "best" or "mean"
    verbose=True,
):
    raw = rawAll.copy()

    if sfreq is None:
        sfreq = raw.info["sfreq"]

    def _default_eye_picks(side):
        side = side.capitalize()
        valid_prefixes = (
            f"{side} Eye ",
            f"{side} Pupil",
        )
        return [ch for ch in raw.ch_names if ch.startswith(valid_prefixes)]

    def _estimate_eye_lag(eye_ref_picks):
        for ch in eye_ref_picks:
            if ch not in raw.ch_names:
                raise RuntimeError(f"Eye reference channel not found: {ch}")

        for k in ("x", "y"):
            if meg_ref_picks[k] not in raw.ch_names:
                raise RuntimeError(
                    f"MEG reference channel not found: {meg_ref_picks[k]}"
                )

        eye_x = raw.get_data(picks=[eye_ref_picks[0]]).squeeze()
        eye_y = raw.get_data(picks=[eye_ref_picks[1]]).squeeze()
        meg_x = raw.get_data(picks=[meg_ref_picks["x"]]).squeeze()
        meg_y = raw.get_data(picks=[meg_ref_picks["y"]]).squeeze()

        lag_x, _, strength_x = measure_lag(
            eye_x, meg_x, sfreq=sfreq, maxlag_sec=maxlag_sec
        )
        lag_y, _, strength_y = measure_lag(
            eye_y, meg_y, sfreq=sfreq, maxlag_sec=maxlag_sec
        )

        if lag_strategy == "best":
            if strength_x >= strength_y:
                return lag_x, strength_x
            else:
                return lag_y, strength_y
        else:
            return (
                int(np.round((lag_x + lag_y) / 2)),
                (strength_x + strength_y) / 2,
            )

    if which_eye == "best":
        lag_L, strength_L = _estimate_eye_lag(left_eye_ref_picks)
        lag_R, strength_R = _estimate_eye_lag(right_eye_ref_picks)

        if strength_L >= strength_R:
            lag = lag_L
            strength = strength_L
            chosen_eye = "left"
            eye_all_picks = (
                left_eye_all_picks
                if left_eye_all_picks is not None
                else _default_eye_picks("left")
            )
        else:
            lag = lag_R
            strength = strength_R
            chosen_eye = "right"
            eye_all_picks = (
                right_eye_all_picks
                if right_eye_all_picks is not None
                else _default_eye_picks("right")
            )
    else:
        chosen_eye = which_eye
        if which_eye == "left":
            lag, strength = _estimate_eye_lag(left_eye_ref_picks)
            eye_all_picks = (
                left_eye_all_picks
                if left_eye_all_picks is not None
                else _default_eye_picks("left")
            )
        elif which_eye == "right":
            lag, strength = _estimate_eye_lag(right_eye_ref_picks)
            eye_all_picks = (
                right_eye_all_picks
                if right_eye_all_picks is not None
                else _default_eye_picks("right")
            )
        else:
            raise ValueError("which_eye must be 'left', 'right', or 'best'")

    if not eye_all_picks:
        raise RuntimeError(
            f"No eye channels found for {chosen_eye} eye. "
            f"Available channels: {raw.ch_names}"
        )

    missing = [ch for ch in eye_all_picks if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f"Eye channels not in raw.info: {missing}")

    if verbose:
        print(
            f"Using {chosen_eye} eye for lag correction: "
            f"{lag} samples ({lag / sfreq:.4f} s), "
            f"corr={strength:.3f}"
        )

    data = raw.get_data()

    picks_eye_all = mne.pick_channels(raw.ch_names, eye_all_picks)
    picks_meg_ref = mne.pick_channels(
        raw.ch_names,
        [meg_ref_picks["x"], meg_ref_picks["y"]],
    )

    eye_data = data[picks_eye_all]
    meg_ref_data = data[picks_meg_ref]

    eye_shifted, meg_ref_shifted = apply_lag_shift_multichan(
        eye_data, meg_ref_data, lag
    )

    n_samples = eye_shifted.shape[1]

    new_data = data[:, :n_samples]
    new_data[picks_eye_all] = eye_shifted
    new_data[picks_meg_ref] = meg_ref_shifted

    raw_aligned = raw.copy().crop(
        tmin=0,
        tmax=(n_samples - 1) / sfreq,
    )
    raw_aligned._data = new_data

    assert raw_aligned._data.shape[0] == len(raw_aligned.ch_names)
    assert raw_aligned._data.shape[1] == raw_aligned.n_times

    return raw_aligned, {
        "lag_samples": lag,
        "lag_seconds": lag / sfreq,
        "strength": strength,
        "chosen_eye": chosen_eye,
        "eye_channels_shifted": eye_all_picks,
        "meg_ref_mapping": meg_ref_picks,
    }
