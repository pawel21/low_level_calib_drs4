# Low level calib DRS4
This is a collection of scripts to create pedestal file.

## To create pedestal file
```source activate cta-dev ```

```python create_pedestal_file.py  --input_file path_to_data --output_file path to pedestal file --max_events maximum number of events to process```

e.g.
``` python create_pedestal_file.py --input_file /20190215/LST-1.\*.Run00097.0000.fits.fz --output_file pedestal97.fits --max_events 9000```