"""
Make check plots for DRS4 calibration run.
"""
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

import sys
import logging
from traitlets import Dict, List, Unicode, Float, Bool, Int


from ctapipe.core import Provenance, traits

from ctapipe.core import Tool
from ctapipe.io import event_source
from ctapipe.io import EventSource

from ctapipe.image.extractor import FixedWindowSum

from lstchain.calib.camera.r0 import LSTR0Corrections

from time_cal_corr import TimeCalCorr
from tools import plot_corr_curve

__all__ = [
    'CalibTime'
]


class CalibTime(Tool):
    """
     Tool that generates a coeff for time correction
     For getting help run:
     python calib_time.py --help
     """

    name = "CalibTime"
    description = "Generate a calib coeff for time correction"

    input_file = Unicode('',
                         help = "Path to LST data"
                        ).tag(config=True)

    n_combine = Int(8,
                    help='Number of combine capacitor???'
                    ).tag(config=True)

    n_harm = Int(16,
                help='Number of harmonic to fourier expansion'
                ).tag(config=True)

    n_cap = Int(1024,
                help='Number of capacitor???'
                ).tag(config=True)

    output_file = Unicode(
        'calibration.hdf5',
        help='Name of the output file'
    ).tag(config=True)


    def setup(self):
        kwargs = dict(parent=self)
        self.log.info("Setup")
        self.eventsource = EventSource.from_config(**kwargs)
        self.extractor = FixedWindowSum(**kwargs)

        self.n = int(self.n_cap / self.n_combine)
        self.timeCorr = TimeCalCorr(self.n_combine, self.n_harm, self.n_cap, offset=400)
        self.lst_r0 = LSTR0Corrections(**kwargs)

        self.log.info(self.n_harm)

        self.log.info(self.extractor.window_start)

    def start(self):
        self.log.info("Start")
        for ev in self.eventsource:
            self.lst_r0.calibrate(ev)

            if ev.r0.event_id % 500 == 0:
                print(ev.r0.event_id)

            self.timeCorr.call_calib_pulse_time_jit(ev)


    def finish(self):
        self.log.warning("Shutting down.")

        self.timeCorr.finalize()

        pdf = matplotlib.backends.backend_pdf.PdfPages("plots/output.pdf")

        for gain in [0, 1]:
            for pix_id in [10, 150, 250, 280, 1000, 1200]:
                self.timeCorr.fit(pix_id, gain=gain)
                an = self.timeCorr.fan
                bn = self.timeCorr.fbn
                fMeanVal = self.timeCorr.fMeanVal[gain, pix_id]
                fig = plot_corr_curve(self.n, self.n_cap, self.n_combine, an, bn, fMeanVal)
                plt.title("Gain = {}, pixel = {}".format(gain, pix_id))
                pdf.savefig(fig)
        pdf.close()

        self.timeCorr.save_to_h5_file(self.output_file)

def main():
    exe = CalibTime()

    exe.run()


if __name__ == '__main__':
    main()
