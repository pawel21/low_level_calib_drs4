import sys
sys.path.insert(0,'/home/pawel1/Pulpit/Astrophysics/CTA/soft/ctapipe_io_lst')
sys.path.insert(0,'/home/pawel1/Pulpit/Astrophysics/CTA/cta-lstchain/lstchain')
from ctapipe.io import event_source
from ctapipe_io_lst import LSTEventSource

import matplotlib.pyplot as plt
import numpy as np

from ctapipe.image.charge_extractors import LocalPeakIntegrator

from calib.camera.r0 import LSTR0Corrections

from tools import get_first_capacitor
from time_cal_corr import TimeCalCorr

plt.rcParams['font.size'] = 20


def corr_time(input_file_1, input_file_2, gain, nr, pix):
    n_combine = 8
    n_harm = 16
    n_cap = 1024
    timeCorr = TimeCalCorr(n_combine, n_harm, n_cap)

    # compute curve to calib time
    reader = event_source(input_url=input_file_1, max_events=4000)
    for ev in reader:
        expected_pixel_id = ev.lst.tel[0].svc.pixel_ids
        fc = get_first_capacitor(ev, nr)
        pixel = expected_pixel_id[nr * 7 + pix]
        integrator = LocalPeakIntegrator(None, None)
        integration, peakpos, window = integrator.extract_charge(ev.r0.tel[0].waveform[:, :, :])
        print("pekpos = {}, charge = {}".format(peakpos[gain, pixel], integration[gain, pixel]))
        if integration[0, pixel] > 4000:
            timeCorr.fill(fc[gain, pix] % 1024, peakpos[gain, pixel])

    timeCorr.finalize()

    fc = np.arange(0, 1024, 8)
    an = timeCorr.fan
    bn = timeCorr.fbn
    y = np.zeros(128)

    for i in range(0, len(y)):
        temp_cos = 0
        temp_sin = 0
        for j in range(0, len(an)):
            temp_cos += an[j] * np.cos(2 * j * np.pi * (fc[i] / 1024.))
            temp_sin += bn[j] * np.sin(2 * j * np.pi * (fc[i] / 1024.))
        y[i] = (temp_cos + temp_sin)

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].plot(np.arange(0, 1024, 8), timeCorr.fMeanVal, 'bo', markersize=4)
    offset = y[0] - timeCorr.fMeanVal[0]
    ax[0].plot(fc, y - offset, 'r--')
    ax[0].set_ylabel("Mean arrival time")
    ax[0].set_xlabel("Position in the DRS ring")
    ax[0].set_title("n_harm = {}, pixel = {}".format(n_harm, pixel))

    # Apply time corr on second file
    reader = event_source(input_url=input_file_2,max_events=4000)
    arrival_time_list = []
    arrival_time_corr_list = []

    for ev in reader:
        expected_pixel_id = ev.lst.tel[0].svc.pixel_ids
        pixel = expected_pixel_id[nr * 7 + pix]
        fc = get_first_capacitor(ev, nr)
        integrator = LocalPeakIntegrator(None, None)
        integration, peakpos, window = integrator.extract_charge(ev.r0.tel[0].waveform[:, :, :])
        print("pekpos = {}, charge = {}".format(peakpos[gain, pixel], integration[gain, pixel]))
        if integration[0, pixel] > 4000:
            arrival_time_list.append(peakpos[gain, pixel])
            arrival_time_corr_list.append(timeCorr.get_corr_time(fc[gain, pix] % 1024))

    binwidth = 1
    ax[1].hist(arrival_time_list, histtype='step', lw=3, linestyle='--', range=(20, 30),
             bins=np.arange(min(arrival_time_list), max(arrival_time_list) + binwidth, binwidth),
             label="before time corr")

    ax[1].hist(arrival_time_corr_list, histtype='step', lw=3, linestyle='--', color='red', range=(20, 30),
             bins=np.arange(min(arrival_time_corr_list), max(arrival_time_corr_list) + binwidth, binwidth),
             label="after time corr")
    ax[1].set_ylabel("Number of events")
    ax[1].set_xlabel("Arrival time [time samples 1 ns]")
    ax[1].legend()
    plt.tight_layout()
    plt.show()


# corr_time("/media/pawel1/ADATA HD330/20190312/LST-1.1.Run00250.0000.fits.fz", "/media/pawel1/ADATA HD330/20190312/LST-1.2.Run00250.0000.fits.fz", 1, 25, 1)