import numpy as np
from matplotlib import pyplot as plt

from ctapipe.core import Tool
from ctapipe.core.traits import Unicode, Integer, Dict, List
from ctapipe.io.lsteventsource import LSTEventSource

from DRS4 import get_first_capacitor_array

plt.rcParams['font.size'] = 25

class MakeHistFirstCap(Tool):
    name = "make-hist-first-cap"
    
    infile = Unicode(
        help='input LST file',
        default=""
    ).tag(config=True)

    max_events = Integer(
        help='stop after this many events if non-zero', default_value=0, min=0
    ).tag(config=True)

    output_suffix = Unicode(
        help='suffix (file extension) of output '
        'filenames to write images '
        'to (no writing is done if blank). '
        'Images will be named [EVENTID][suffix]',
        default_value=""
    ).tag(config=True)

    aliases = Dict({
        'infile': 'MakeHistFirstCap.infile',
        'max-events': 'MakeHistFirstCap.max_events',
        'output-suffix': 'MakeHistFirstCap.output_suffix'
    })


    def setup(self):
        # load LST data
        self.log.info("Read file:{}".format(self.infile))
        self.reader = LSTEventSource(
            input_url=self.infile, max_events=self.max_events
        )

    def start(self):
        fc = []
        for event in self.reader:
            fc.extend(get_first_capacitor_array(event))

        plt.figure()
        plt.hist(fc, bins=4096)
        plt.ylabel("Number")
        plt.xlabel("Stop Cell")
        plt.show()


if __name__ == '__main__':
    tool = MakeHistFirstCap()
    tool.run()
