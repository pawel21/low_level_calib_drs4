import numpy as np


def extract_pulse_time(waveforms):
    window_shift = 3
    window_width = 7
    peak = waveforms.argmax(1)
    start = peak - window_shift
    end = start + window_width
    ind = np.indices(waveforms.shape)[1]
    integration_window = ((ind >= start[..., np.newaxis]) &
                        (ind < end[..., np.newaxis]))
    samples_i = np.indices(waveforms.shape)[1]
    pulse_time = np.average(samples_i, weights=waveforms*integration_window, axis=1)
    outside = np.logical_or(pulse_time < 0, pulse_time >= waveforms.shape[1])
    pulse_time[outside] = -1
    return pulse_time
