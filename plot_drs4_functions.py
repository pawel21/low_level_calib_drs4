import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_waveform(event, gain, nr_clus, pixel, ylim=[200, 400]):
    time = np.linspace(2, 37, 36)
    plt.figure(figsize=(8,6))
    plt.step(time, event.r0.tel[0].waveform[gain, pixel + nr_clus * 7, 2:38], 'b-', label="R0", lw=2)
    plt.step(time, event.r1.tel[0].waveform[gain, pixel + nr_clus * 7, 2:38], 'r-', label="R1", lw=4)
    plt.plot([0, 40], [300, 300], 'g--', label="offset", lw=3)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel("sample")
    plt.ylabel("signal [counts]")
    plt.grid(True)
    plt.show()


def plot_hist_all_pixels(ev_r0_array, ev_r1_array, nr_clus):
    position = [(0, 0), (0, 1), (0, 2), (0, 3),
                (1, 0), (1, 1), (1, 2), (1, 3)]
    high_gain = 0
    low_gain = 1

    fig, ax = plt.subplots(2, 4, figsize=(18, 11))
    for i in range(0, 7):
        ax[position[i]].hist(ev_r0_array[nr_clus, :, high_gain, i, 2:38].ravel(), bins=50, facecolor='blue',
                             histtype='stepfilled', range=(100, 400))
        ax[position[i]].hist(ev_r1_array[nr_clus, :, high_gain, i, 2:38].ravel(), bins=50, facecolor='red',
                             histtype='stepfilled', range=(100, 400), alpha=0.7)
        ax[position[i]].set_xlabel("signal [counts]")
        ax[position[i]].set_ylabel("number of events")
        ax[position[i]].set_xlim([100, 400])
        ax[position[i]].set_yscale('log')
        ax[position[i]].set_title("HG pixel = {}".format(i))
        ax[position[7]].axis('off')
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(18, 11))
    for i in range(0, 7):
        ax[position[i]].hist(ev_r0_array[nr_clus, :, low_gain, i, 2:38].ravel(), bins=50, facecolor='blue',
                             histtype='stepfilled', range=(200, 400))
        ax[position[i]].hist(ev_r1_array[nr_clus, :, low_gain, i, 2:38].ravel(), bins=50, facecolor='red',
                             histtype='stepfilled', range=(200, 400), alpha=0.7)
        ax[position[i]].set_xlabel("signal [counts]")
        ax[position[i]].set_ylabel("number of events")
        ax[position[i]].set_xlim([100, 400])
        ax[position[i]].set_yscale('log')
        ax[position[i]].set_title("LG pixel = {}".format(i))
        ax[position[7]].axis('off')
    plt.show()