import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import optimize
from numba import njit, prange

from matplotlib.colors import LogNorm


class DtCurveFit:
    def __init__(self):
        self.last_time_array = np.zeros((7, 4096))
        self.size4drs = 4096
        self.dt = [[] for i in range(7)]
        self.baseline = [[] for i in range(7)]

    def fill(self, event, gain, nr_module):
        expected_pixel_id = event.lst.tel[0].svc.pixel_ids
        local_clock_list = event.lst.tel[0].evt.local_clock_counter

        time_now = local_clock_list[nr_module]
        fc = get_first_capacitor(event, nr_module)

        for pix in prange(0, 7):
            pixel = expected_pixel_id[nr_module * 7 + pix]
            for k in prange(0, 40):
                posads = int((k + fc[gain, pix]) % self.size4drs)
                if self.last_time_array[pix, posads] > 0:
                    time_diff = time_now - self.last_time_array[pix, posads]
                    if k > 2 and k < 38:
                        self.dt[pix].append(time_diff)
                        self.baseline[pix].append(event.r1.tel[0].waveform[gain, pixel, k])
                if (k < 39):
                    self.last_time_array[pix, posads] = time_now
                if pix % 2 == 0:
                    first_cap = int(fc[gain, pix])
                    if first_cap % 1024 > 766 and first_cap % 1024 < 1012:
                        start = int(first_cap) + 1024 - 1
                        end = int(first_cap) + 1024 + 11
                        self.last_time_array[pix, start % 4096:end % 4096] = time_now
                    elif first_cap % 1024 >= 1012:
                        channel = int(first_cap / 1024)
                        for kk in range(first_cap + 1024, (channel + 2) * 1024):
                            self.last_time_array[pix, int(kk) % 4096] = time_now


def get_first_capacitor(event, nr_module):
    """
    Get first capacitor values from event for nr module.
    Parameters
    ----------
    event : `ctapipe` event-container
    nr_module : number of module
    """
    high_gain = 0
    low_gain = 1
    fc = np.zeros((2, 7))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr_module * 8:
                                                        (nr_module + 1) * 8]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[high_gain, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[low_gain, i] = first_cap[j]
    return fc
