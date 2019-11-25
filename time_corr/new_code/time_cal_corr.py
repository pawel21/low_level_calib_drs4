import numpy as np
from ctapipe.core.traits import Int
from numba import njit, prange

from ctapipe.image.extractor import LocalPeakWindowSum


class TimeCalCorr:
    high_gain = 0
    low_gain = 1

    def __init__(self, n_combine, n_harm, n_cap, tel_id=1, offset=400, window_width=7, window_shift=3):
        n = int(n_cap/n_combine)
        self.fMeanVal = np.zeros((2, 1855, n))
        self.fNumMean = np.zeros((2, 1855, n))
        self.fNumCombine = n_combine
        self.fNumHarmonics = n_harm
        self.fNumCap = n_cap
        self.fNumPoints = int(self.fNumCap/self.fNumCombine)
        self.tel_id = tel_id
        self.offset = offset
        self.first_cap_array = np.zeros((265, 2, 7))
        self.n_modules = 265
        self.extractor = LocalPeakWindowSum(window_width=window_width, window_shift=window_shift)

    def calib_pulse_time(self, ev):
        pixel_ids = ev.lst.tel[self.tel_id].svc.pixel_ids
        baseline_subtracted = ev.r1.tel[self.tel_id].waveform[:, :, 2:38] - self.offset
        charge, pulse_time = self.extractor(baseline_subtracted)

        for gain in prange(0, 2):
            for nr in prange(0, 265):
                fc = self.get_first_capacitor(ev, nr)
                for pix in prange(0, 7):
                    pixel = pixel_ids[nr * 7 + pix]
                    if ev.r0.tel[self.tel_id].trigger_type == 1 and np.mean(ev.r1.tel[self.tel_id].waveform[0, pixel, 2:38]) > 100:
                        first_cap = (fc[gain, pix]) % self.fNumCap
                        fBin = int(first_cap / self.fNumCombine)
                        self.fMeanVal[gain, pixel, fBin] += pulse_time[gain, pixel]
                        self.fNumMean[gain, pixel, fBin] += 1

    def finalize(self):
        self.fMeanVal = self.fMeanVal /self.fNumMean

    def fit(self, pixel_id, gain=0):
        self.pos = np.zeros(self.fNumPoints)
        for i in range(0, self.fNumPoints):
            self.pos[i] = ( i +0.5 ) *self.fNumCombine

        self.fan = np.zeros(self.fNumHarmonics)
        self.fbn = np.zeros(self.fNumHarmonics)

        for n in range(0, self.fNumHarmonics):
            self.integrate_with_trig(self.pos, self.fMeanVal[gain, pixel_id], n, self.fan, self.fbn)

    def integrate_with_trig(self, x, y, n, an, bn):
        suma = 0
        sumb = 0

        for i in range(0, self.fNumPoints):
            suma += y[i] *self.fNumCombine *np.cos( 2 *np.pi * n *(x[i] /float(self.fNumCap)))
            sumb += y[i] *self.fNumCombine *np.sin( 2 *np.pi * n *(x[i] /float(self.fNumCap)))

        an[n] = suma *(2./(self.fNumPoints *self.fNumCombine))
        bn[n] = sumb *(2./(self.fNumPoints *self.fNumCombine))

    def get_first_capacitor(self, event, nr):
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[self.tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        # First capacitor order according Dragon v5 board data format
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc


def get_corr_time(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time



