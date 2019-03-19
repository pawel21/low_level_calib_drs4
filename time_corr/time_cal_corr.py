import numpy as np


class TimeCalCorr:
    def __init__(self, n_combine, n_harm, n_cap):
        self.fMeanVal = np.zeros(128)
        self.fNumMean = np.zeros(128)
        self.fNumCombine = n_combine
        self.fNumHarmonics = n_harm
        self.fNumCap = n_cap
        self.fNumPoints = int((self.fNumCap ) /(self.fNumCombine))

    def fill(self, first_cap, arrslice):
        fBin = int(first_cap /self.fNumCombine)
        self.fMeanVal[fBin] += arrslice
        self.fNumMean[fBin] += 1

    def finalize(self):
        self.fMeanVal = self.fMeanVal /self.fNumMean

        self.pos = np.zeros(self.fNumPoints)
        for i in range(0, self.fNumPoints):
            self.pos[i] = ( i +0.5 ) *self.fNumCombine

        self.fan = np.zeros(self.fNumHarmonics)
        self.fbn = np.zeros(self.fNumHarmonics)

        for n in range(0, self.fNumHarmonics):
            self.integrate_with_trig(self.pos, self.fMeanVal, n, self.fan, self.fbn)

    def integrate_with_trig(self, x, y, n, an, bn):
        suma = 0
        sumb = 0

        for i in range(0, self.fNumPoints):
            suma += y[i] *self.fNumCombine *np.cos( 2 *np.pi * n *(x[i] /float(self.fNumCap)))
            sumb += y[i] *self.fNumCombine *np.sin( 2 *np.pi * n *(x[i] /float(self.fNumCap)))

        an[n] = suma *(2./(self.fNumPoints *self.fNumCombine))
        bn[n] = sumb *(2./(self.fNumPoints *self.fNumCombine))

    def get_corr_time(self, first_cap):
        time = self.fan[0] / 2.
        for n in range(1, self.fNumHarmonics):
            time += self.fan[n] * np.cos((first_cap * n * 2 * np.pi) / self.fNumCap)
            time += self.fbn[n] * np.sin((first_cap * n * 2 * np.pi) / self.fNumCap)
        return time
