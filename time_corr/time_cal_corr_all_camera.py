import numpy as np


class TimeCalCorr:
    def __init__(self, n_combine, n_harm, n_cap):
        n = int(n_cap/n_combine)
        self.fMeanVal = np.zeros((1855, n))
        self.fNumMean = np.zeros((1855, n))
        self.fNumCombine = n_combine
        self.fNumHarmonics = n_harm
        self.fNumCap = n_cap
        self.fNumPoints = int(self.fNumCap/self.fNumCombine)

    def fill(self, gain, pixel_ids, first_cap_array, integration, pulse_time):
        for nr in range(0, 265):
            fc = get_first_capacitor(first_cap_array, nr)
            for pix in range(0, 7):
                pixel = pixel_ids[nr*7 + pix]
                if integration[gain, pixel] > 3000:
                    first_cap = fc[gain, pix]%1024
                    fBin = int(first_cap /self.fNumCombine)
                    self.fMeanVal[pixel, fBin] += pulse_time[pixel]
                    self.fNumMean[pixel, fBin] += 1

    def finalize(self):
        self.fMeanVal = self.fMeanVal /self.fNumMean

    def fit(self, pixel_id):
        self.pos = np.zeros(self.fNumPoints)
        for i in range(0, self.fNumPoints):
            self.pos[i] = ( i +0.5 ) *self.fNumCombine

        self.fan = np.zeros(self.fNumHarmonics)
        self.fbn = np.zeros(self.fNumHarmonics)

        for n in range(0, self.fNumHarmonics):
            self.integrate_with_trig(self.pos, self.fMeanVal[pixel_id], n, self.fan, self.fbn)

    def integrate_with_trig(self, x, y, n, an, bn):
        suma = 0
        sumb = 0

        for i in range(0, self.fNumPoints):
            suma += y[i] *self.fNumCombine *np.cos( 2 *np.pi * n *(x[i] /float(self.fNumCap)))
            sumb += y[i] *self.fNumCombine *np.sin( 2 *np.pi * n *(x[i] /float(self.fNumCap)))

        an[n] = suma *(2./(self.fNumPoints *self.fNumCombine))
        bn[n] = sumb *(2./(self.fNumPoints *self.fNumCombine))


def get_corr_time(first_cap, fan, fbn, fNumHarmonics=16, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time

def get_first_capacitor(first_capacitor_id, nr_module):
    fc = np.zeros((2, 7))
    first_cap = first_capacitor_id[nr_module*8:(nr_module + 1) * 8]
    high_gain = 0
    low_gain = 1
    # First capacitor order according Dragon v5 board data format
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[high_gain, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[low_gain, i] = first_cap[j]
    return fc
