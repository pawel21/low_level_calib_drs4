import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import optimize
from numba import njit, prange

from matplotlib.colors import LogNorm


class DtCorrFit:
    def __init__(self):
        self.last_time_array = np.zeros((7, 4096))
        self.size4drs = 4096
        self.dt = [[] for i in range(7)]
        self.baseline = [[] for i in range(7)]
        self.baseline_corr = [[] for i in range(7)]

    def calib(self, event, gain, nr_module):
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
                        self.dt[pix].append(time_diff / 133.e3)
                        self.baseline[pix].append(event.r1.tel[0].waveform[gain, pixel, k])
                        val = event.r1.tel[0].waveform[gain, pixel, k] - ped_time(time_diff / (133.e3))
                        self.baseline_corr[pix].append(val)
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


def ped_time(timediff):
    """
    Power law function for time lapse baseline correction.
    Coefficients from curve fitting to dragon test data
    at temperature 40 degC
    """
    return 23.03 * np.power(timediff, -0.25) - 9.73


def plot_dt_curve(dt, baseline, baseline_corr):
    t = np.linspace(0.01, 800, 13000)
    y = (23.03 * np.power(t, -0.25) - 9.73) + 300

    p0 = [29., -0.2, -12.]  # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(dt, np.array(baseline)))
    Y = power_law(p1, t)
    print(p1)

    fig, ax = plt.subplots(2, 1, figsize=(18, 16))
    counts, xedges, yedges, im = ax[0].hist2d(np.log10(dt), baseline, bins=50, norm=LogNorm(), cmap=plt.cm.rainbow)
    ax[0].plot(np.log10(t), y, 'k--', lw=5, label="$y = A \cdot dt^B + C$")
    ax[0].plot(np.log10(t), Y, 'r--', lw=5, label="fit")
    ax[0].plot([-3, 40], [300, 300], 'g--', lw=5)
    ax[0].set_xticks([-2, -1, 0, 1, 2], ['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    ax[0].set_xticklabels(['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])

    ax[0].set_xlabel("Różnica czasu odczytu kondensatora [ms], $dt$")
    ax[0].set_ylabel("Syganł [zliczenia ADC]")
    ax[0].set_ylim([255, 400])
    ax[0].legend()
    plt.colorbar(im, ax=ax[0], label="zliczenia")

    counts, xedges, yedges, im = ax[1].hist2d(np.log10(dt), baseline_corr, bins=50, norm=LogNorm(), cmap=plt.cm.rainbow)
    ax[1].plot([-3, 40], [300, 300], 'g--', lw=5)
    ax[1].set_xticks([-2, -1, 0, 1, 2], ['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    plt.colorbar(im, ax=ax[1], label="zliczenia")
    ax[1].set_xlabel("Różnica czasu odczytu kondensatora [ms], $dt$")
    ax[1].set_ylabel("Syganł [zliczenia ADC]")
    ax[1].set_xticks([-2, -1, 0, 1, 2], ['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    ax[1].set_xticklabels(['$10^{-2}$', '$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    ax[1].set_ylim([255, 400])
    plt.tight_layout()
    plt.show()


power_law = lambda p, dt: p[0] * dt ** p[1] + p[2]
errfunc = lambda p, x, y: power_law(p, x) - y