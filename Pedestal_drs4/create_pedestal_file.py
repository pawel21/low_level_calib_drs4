import argparse
import numpy as np
from astropy.io import fits
from ctapipe.io import EventSeeker

import sys
sys.path.append('/home/pawel1/Pulpit/Astrophysics/CTA/cta-lstchain')
from lstchain.io.lsteventsource import LSTEventSource

from drs4 import DragonPedestal
from r0 import LSTR0Corrections

parser = argparse.ArgumentParser()


parser.add_argument("--input_file", help="Path to fitz.fz file to create pedestal file.",
                    type=str, default="")

parser.add_argument("--output_file", help="Path where script create pedestal file",
                    type=str)

parser.add_argument("--max_events", help="Maximum numbers of events to read",
                    type=int, default=5000)

args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    print("max events: {}".format(args.max_events))
    reader = LSTEventSource(input_url=args.input_file, max_events=args.max_events)
    print("---> Number of files", reader.multi_file.num_inputs())

    lst_r0 = LSTR0Corrections()

    seeker = EventSeeker(reader)
    ev = seeker[0]
    n_modules = ev.lst.tel[0].svc.num_modules

    ped = DragonPedestal()
    PedList = []
    pedestal_value_array = np.zeros((n_modules, 2, 7, 4096), dtype=np.uint16)

    for i in range(0, n_modules):
        PedList.append(DragonPedestal())

    for i, event in enumerate(reader):
        print("i = {}, ev id = {}".format(i, event.r0.event_id))
        lst_r0.time_lapse_corr(ev)
        lst_r0.interpolate_spikes(ev)
        for nr_module in range(0, n_modules):
            PedList[nr_module].fill_pedestal_event(event, nr=nr_module)

    # Finalize pedestal and write to fits file
    for i in range(0, n_modules):
        PedList[i].finalize_pedestal()
        PedList[i].meanped[np.isnan(PedList[i].meanped)] = 300  # fill 300 where occurs NaN
        pedestal_value_array[i, :, :, :] = PedList[i].meanped

    primaryhdu = fits.PrimaryHDU(ev.lst.tel[0].svc.pixel_ids)
    secondhdu = fits.ImageHDU(pedestal_value_array)

    hdulist = fits.HDUList([primaryhdu, secondhdu])
    hdulist.writeto(args.output_file)
