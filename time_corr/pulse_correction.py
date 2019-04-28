import numpy as np
from numba import njit, prange

from ctapipe.image.extractor import LocalPeakWindowSum
from simple_extracor import extract_pulse_time
from tools import get_first_capacitor

class PulseCorrection:

    extractor = LocalPeakWindowSum()

    def __init__(self, fan_array, fbn_array, n_harm, offset=380):
        self.fan_array = fan_array
        self.fbn_array = fbn_array
        self.n_harm = n_harm
        self.offset = offset
        self.mean_time_list = []
        self.raw_pulse_list = [[] for i in range(1855)]
        self.corr_pulse_list = [[] for i in range(1855)]
        self.corr_mean_pulse_list = [[] for i in range(1855)]


    def corr_pulse(self, ev):
        expected_pixel_id = ev.lst.tel[0].svc.pixel_ids
        baseline_subtracted = ev.r1.tel[0].waveform[:, :, 2:38] - 380
        try:
            charge, _ = self.extractor(baseline_subtracted)
            pulse_time =  extract_pulse_time(baseline_subtracted[0, :, :])
            corr_time_list = []
            for nr in prange(0, 265):
                fc = get_first_capacitor(ev, nr)
                for pix in prange(0, 7):
                    pixel = expected_pixel_id[nr*7 + pix]

                    if charge[0, pixel] > 1500:
                        corr_pos = get_corr_time(fc[0, pix]%1024, self.fan_array[pixel], self.fbn_array[pixel],
                                                            fNumHarmonics=self.n_harm)
                        corr_time = pulse_time[pixel] - corr_pos
                        corr_time_list.append(corr_time)
            if len(corr_time_list) > 1500:
                mean_time = np.nanmean(corr_time_list)
                self.mean_time_list.append(mean_time)

                for nr in prange(0, 265):
                    fc = get_first_capacitor(ev, nr)
                    for pix in prange(0, 7):
                        pixel = expected_pixel_id[nr*7 + pix]

                        if charge[0, pixel] > 1500:
                            corr_pos = get_corr_time(fc[0, pix]%1024, self.fan_array[pixel], self.fbn_array[pixel],
                                                            fNumHarmonics=self.n_harm)
                            corr_time = pulse_time[pixel] - corr_pos
                            dt = corr_time - mean_time

                            self.raw_pulse_list[pixel].append(pulse_time[pixel])
                            self.corr_pulse_list[pixel].append(corr_time)
                            self.corr_mean_pulse_list[pixel].append(dt)
        except Exception as err:
            print(err)

    def get_raw_pulse_list(self):
        return self.raw_pulse_list

    def get_corr_pulse_list(self):
        return self.corr_pulse_list

    def get_corr_mean_pulse_list(self):
        return self.corr_mean_pulse_list

def get_corr_time(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time
