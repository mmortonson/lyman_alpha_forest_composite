## Simulating the composite spectrum of the Lyman alpha forest

This repo contains code for computing the composite (basically, averaged) 
spectrum from simulated observations of the Lyman alpha forest in quasar 
spectra. Details on the science behind this project can be found at [1].

The main files are:

- `sim_spec_lya.py`: take data from skewers through cosmological simulations 
  		     and smooth, pixelize, and add simulated noise to
		     approximate SDSS-III/BOSS spectra
- `stack_lya.py`: combine multiple simulated spectra to compute a composite
  		  spectrum; this process reduces noise and makes weak 
		  features in the data easier to detect

[1]: http://arxiv.org/abs/1309.6768
