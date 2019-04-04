import matplotlib.pyplot as plt
import numpy as np


class TimeCalCorr:
    def __init__(self, n_combine, n_harm, n_cap):
        n = int(n_cap/n_combine)
        self.fMeanVal = np.zeros(n)
        self.fNumMean = np.zeros(n)
        self.fRms = np.zeros(n)
        self.fNumCombine = n_combine
        self.fNumHarmonics = n_harm
        self.fNumCap = n_cap
        self.fNumPoints = int(self.fNumCap/self.fNumCombine)

    def fill(self, first_cap, arrslice):
        fBin = int(first_cap /self.fNumCombine)
        self.fMeanVal[fBin] += arrslice
        self.fNumMean[fBin] += 1
        self.fRms[fBin] += arrslice * arrslice

    def finalize(self):
        self.fMeanVal = self.fMeanVal /self.fNumMean
        self.fRms = np.sqrt((self.fRms / self.fNumMean) - self.fMeanVal ** 2)

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


def plot_corr_curve(n, n_cap, n_combine, an, bn, fMeanVal):
    fc = np.arange(0, n_cap, n_combine)
    y = np.zeros(n)

    for i in range(0, len(y)):
        temp_cos = an[0] / 2
        temp_sin = 0
        for j in range(1, len(an)):
            temp_cos += an[j] * np.cos(2 * j * np.pi * (fc[i] / 1024.))
            temp_sin += bn[j] * np.sin(2 * j * np.pi * (fc[i] / 1024.))
        y[i] = (temp_cos + temp_sin)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(np.arange(0, n_cap, n_combine), fMeanVal, 'bo')
    ax.plot(fc, y, 'r--')
    ax.set_ylabel("Mean arrival time")
    ax.set_xlabel("Position in the DRS ring")


def make_hist(arrival_time_list, arrival_time_corr_list, binwidth = 0.5):
    plt.figure(figsize=(16, 9))
    plt.hist(arrival_time_list, histtype='step', lw=3, linestyle='--', range=(20, 30),
             bins=np.arange(min(arrival_time_list), max(arrival_time_list) + binwidth, binwidth),
             label="before time corr")

    plt.hist(arrival_time_corr_list, histtype='step', lw=3, linestyle='--', color='red', range=(20, 30),
             bins=np.arange(min(arrival_time_corr_list), max(arrival_time_corr_list) + binwidth, binwidth),
             label="after time corr")
    plt.ylabel("Number of events")
    plt.xlabel("Arrival time [time samples 1 ns]")
    plt.legend()
    plt.tight_layout()