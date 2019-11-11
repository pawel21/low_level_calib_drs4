import argparse
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.utils import Histogram
from ctapipe_io_lst import LSTEventSource

plt.rcParams['font.size'] = 15

'''
 This is script to check DRS4 run with correction applied online by EVB3.
 To run script:
 python check_drs4_calib_onsite.py --input_file LST-1.1.Run01578.0000.fits.fz --output_file plot.png
 Script produce histogram and calcurate RMS (which should be below 10 ADC) for both gains.
'''


def get_mean_rms(hist):
    n =  hist.data
    bins = hist.bin_lower_edges
    mids = 0.5*(bins[0][1:] + bins[0][:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean)**2, weights=n)
    return mean, np.sqrt(var)


parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file",
                    help="Path to DRS4 run after pedestal subtraction.",
                    type=str,
                    default="")

parser.add_argument("--output_file",
                    help="Path to output plot in png format.",
                    type=str,
                    default="")

args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    run_path = args.input_file

    print("Run path: {}".format(run_path))

    reader = LSTEventSource(input_url=run_path)

    hist_r0_hg = Histogram(ranges=[0, 1000], nbins=(1000))
    hist_r0_lg = Histogram(ranges=[0, 1000], nbins=(1000))

    for i, ev in enumerate(reader):
        if i%5 == 0: # histogram is filled every 5 event
            hist_r0_hg.fill(ev.r0.tel[1].waveform[0, :, 2:38].ravel())
            hist_r0_lg.fill(ev.r0.tel[1].waveform[1, :, 2:38].ravel())

    mean_hg, rms_hg = get_mean_rms(hist_r0_hg)
    mean_lg, rms_lg = get_mean_rms(hist_r0_lg)

    print("RMS HG = {:.2f} ADC".format(rms_hg))
    print("RMS LG = {:.2f} ADC".format(rms_lg))

    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    hist_r0_hg.draw_1d()
    plt.yscale('log')
    plt.xlabel("Signal [ADC]")
    plt.ylabel("Number of samples")
    plt.title("High gain: \n Mean = {:.2f} ADC, RMS = {:.2f} ADC".format(mean_hg, rms_hg))

    plt.subplot(122)
    hist_r0_lg.draw_1d()
    plt.yscale('log')
    plt.xlabel("Signal [ADC]")
    plt.ylabel("Number of samples")
    plt.title("Low gain: \n Mean = {:.2f} ADC, RMS = {:.2f} ADC".format(mean_lg, rms_lg))

    plt.tight_layout()
    plt.savefig(args.output_file)
