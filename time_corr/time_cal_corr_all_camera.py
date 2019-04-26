import numpy as np
from numba import njit, prange

from ctapipe.image.extractor import LocalPeakWindowSum

class TimeCalCorr:

    extractor = LocalPeakWindowSum()

    def __init__(self, n_combine, n_harm, n_cap):
        n = int(n_cap/n_combine)
        self.fMeanVal = np.zeros((1855, n))
        self.fNumMean = np.zeros((1855, n))
        self.fNumCombine = n_combine
        self.fNumHarmonics = n_harm
        self.fNumCap = n_cap
        self.fNumPoints = int(self.fNumCap/self.fNumCombine)

        self.first_cap_array = np.zeros((265, 2, 7))
        self.n_modules = 265

    def calib_pulse_time(self, ev):
        pixel_ids = ev.lst.tel[0].svc.pixel_ids
        get_first_capacitor_jit(ev.lst.tel[0].evt.first_capacitor_id, self.first_cap_array)

        try:
            baseline_subtracted = ev.r1.tel[0].waveform[:, :, 2:38] - 380
            charge, _ = self.extractor(baseline_subtracted)
            pulse_time =  extract_pulse_time(baseline_subtracted[0, :, :]) + 2
            fill_jit(0, pixel_ids, self.first_cap_array, charge, pulse_time,
                self.fMeanVal, self.fNumMean, fNumCombine=self.fNumCombine)
        except ZeroDivisionError:
            pass

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

@njit(fastmath=True, parallel=True)
def fill_jit(gain, pixel_ids, first_cap_array, charge, pulse_time, fMeanVal, fNumMean, fNumCombine=8):
    for nr in prange(0, 265):
        fc =  first_cap_array[nr, :, :]
        for pix in prange(0, 7):
            pixel = pixel_ids[nr*7 + pix]
            if charge[gain, pixel] > 1500:
                first_cap = (fc[gain, pix])%1024
                fBin = int(first_cap /fNumCombine)
                fMeanVal[pixel, fBin] += pulse_time[pixel]
                fNumMean[pixel, fBin] += 1


@njit(fastmath=True, parallel=True)
def get_first_capacitor_jit(event_fc, first_cap_array):
    for nr_module in prange(0, 265):
        first_cap_array[nr_module, :, :] = get_first_capacitor(event_fc, nr_module)

@njit(fastmath=True)
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


class ArrivalTimeCorr:

    extractor = LocalPeakWindowSum()

    def __init__(self, fan_array, fbn_array, n_harm, offset=380):
        self.fan_array = fan_array
        self.fbn_array = fbn_array
        self.n_harm = n_harm
        self.offset = offset
        self.arrival_time_list = [[] for i in range(1855)]
        self.arrival_time_corr_list = [[] for i in range(1855)]

    def corr_arrivial_time(self, ev, N_module=265):
        expected_pixel_id = ev.lst.tel[0].svc.pixel_ids
        baseline_subtracted = ev.r1.tel[0].waveform[:, :, 2:38] - 380
        try:
            charge, pulse_time = self.extractor(baseline_subtracted)
            pulse_time = extract_pulse_time(baseline_subtracted[0, :, :]) + 2
            for nr in prange(0, N_module):
                fc = get_first_capacitor(ev.lst.tel[0].evt.first_capacitor_id, nr)
                for pix in prange(0, 7):
                    pixel = expected_pixel_id[nr * 7 + pix]
                    if charge[0, pixel] > 1500:
                        self.arrival_time_list[pixel].append(pulse_time[pixel])
                        corr_pos = get_corr_time(fc[0, pix]%1024, self.fan_array[pixel], self.fbn_array[pixel],
                                            fNumHarmonics=self.n_harm)
                        corr_time = pulse_time[pixel] - corr_pos #+ get_mean_time(self.fan_array[pixel])
                        self.arrival_time_corr_list[pixel].append(corr_time)
        except ZeroDivisionError:
            pass

    def get_arrivial_time_list(self):
        return self.arrival_time_list

    def get_arrivial_time_corr_list(self):
        return self.arrival_time_corr_list


def get_corr_time(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time


def get_mean_time(fan):
    mean_time = fan[0] / 2.
    return mean_time

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
    print("error = {}".format(mean_squared_error(fMeanVal, y)))

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
