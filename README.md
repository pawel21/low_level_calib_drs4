# Low level calib DRS4
This is a collection of scripts to low level calibration DRS4 data.

## To create binary file to store baseline value of each capacitor
```source activate cta-dev ```

```python tools/create_binary_file_with_pedestals.py  path_to_data output_file```

e.g.
``` python tools/create_binary_file_with_pedestals.py /media/pawel1/ADATA\ HD330/20181006/LST-1.Run0009.0000.fits.fz pedestal_0009.dat ```