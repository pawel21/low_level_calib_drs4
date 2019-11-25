import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from ctapipe.image.extractor import LocalPeakWindowSum


def plot_corr_curve(n, n_cap, n_combine, an, bn, fMeanVal):
    fc = np.arange(0, n_cap, n_combine)
    y = np.zeros(n)

    for i in range(0, len(y)):
        temp_cos = an[0] / 2
        temp_sin = 0
        for j in range(1, len(an)):
            temp_cos += an[j] * np.cos(2 * j * np.pi * (fc[i] / n_cap))
            temp_sin += bn[j] * np.sin(2 * j * np.pi * (fc[i] / n_cap))
        y[i] = (temp_cos + temp_sin)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(np.arange(0, n_cap, n_combine), fMeanVal, 'bo')
    ax.plot(fc, y, 'r--')
    ax.set_ylabel("Mean arrival time")
    ax.set_xlabel("Position in the DRS ring")

def format_axes(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    #ax.set_title("")
    return ax

def get_corr_time(first_cap, fan, fbn, fNumHarmonics, fNumCap=1024):
    time = fan[0] / 2.
    for n in range(1, fNumHarmonics):
        time += fan[n] * np.cos((first_cap * n * 2 * np.pi) / fNumCap)
        time += fbn[n] * np.sin((first_cap * n * 2 * np.pi) / fNumCap)
    return time


def get_first_capacitor(event, nr, tel_id):
    high_gain = 0
    low_gain = 1
    fc = np.zeros((2, 7))
    first_cap = event.lst.tel[tel_id].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
    # First capacitor order according Dragon v5 board data format
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[high_gain, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[low_gain, i] = first_cap[j]
    return fc

extractor = LocalPeakWindowSum()

def get_pulse_before_and_after_time_corr(event):
    gain = 0
    tel_id = 1
    n_harm = 16
    pixel_ids = ev.lst.tel[tel_id].svc.pixel_ids

    time_list = []
    time_corr_list = []
    time_corr_mean_list = []

    for nr in range(0, 265):
        for pix in range(0, 7):
            fc = get_first_capacitor(event, nr, tel_id)[gain, pix]
            pixel = pixel_ids[nr * 7 + pix]

            time = pulse_time[gain, pixel]
            time_corr = time - get_corr_time(fc%1024, fan_array[pixel], fbn_array[pixel], n_harm)
            time_corr_mean = time - get_corr_time(fc%1024, fan_array[pixel], fbn_array[pixel], n_harm) + np.mean(pulse_time[gain, :])

            time_list.append(time)
            time_corr_list.append(time_corr)
            time_corr_mean_list.append(time_corr_mean)
