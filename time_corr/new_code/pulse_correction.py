import h5py
import numpy as np
from numba import njit, prange

from ctapipe.core.traits import Int
from ctapipe.image.extractor import LocalPeakWindowSum

from tools import get_corr_time

class PulseCorrection:
    high_gain = 0
    low_gain = 1

    def __init__(self, n_harm, calib_file_path, tel_id=1, window_width=7, window_shift=3):
        self.fNumHarmonics = n_harm
        self.tel_id = tel_id
        self.n_modules = 265
        self.extractor = LocalPeakWindowSum(window_width=window_width,
                                            window_shift=window_shift)
        self.calib_file_path = calib_file_path
        self.fan_array = None
        self.fbn_array = None

        self.load_calib_file()

    def load_calib_file(self):
        hf = h5py.File(self.calib_file_path, 'r')
        fan = hf.get('fan')
        self.fan_array = np.array(fan)
        fbn = hf.get('fbn')
        self.fbn_array = np.array(fbn)

    def get_pulse_before_and_after_time_corr(self, event):
        gain = 0
        pixel_ids = event.lst.tel[self.tel_id].svc.pixel_ids

        time_list = []
        time_corr_list = []
        time_corr_mean_list = []
        time_corr_relative_list = []

        charge, pulse_time = self.extractor(event.r1.tel[self.tel_id].waveform[:, :, 2:38])

        for nr in range(0, 265):
            for pix in range(0, 7):
                fc = self.get_first_capacitor(event, nr)[gain, pix]
                pixel = pixel_ids[nr * 7 + pix]

                time = pulse_time[gain, pixel]
                time_corr = time - get_corr_time(fc%1024, self.fan_array[pixel], self.fbn_array[pixel], self.fNumHarmonics)
                time_corr_mean = time - get_corr_time(fc%1024, self.fan_array[pixel], self.fbn_array[pixel], self.fNumHarmonics) + np.mean(pulse_time[gain, :])
                time_corr_relative = time - get_corr_time(fc%1024, self.fan_array[pixel], self.fbn_array[pixel], self.fNumHarmonics) - np.mean(pulse_time[gain, :])

                time_list.append(time)
                time_corr_list.append(time_corr)
                time_corr_mean_list.append(time_corr_mean)
                time_corr_relative_list.append(time_corr_relative)

        return time_list, time_corr_list, time_corr_mean_list, time_corr_relative_list


    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc
