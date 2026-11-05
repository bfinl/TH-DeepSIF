# forward/

Training-data generation: neural mass model (NMM) simulation of cortical source
dynamics and projection through a head model to scalp EEG.

## Provenance

**This pipeline is not original to TH-DeepSIF.** It is derived from the DeepSIF
project (https://github.com/bfinl/DeepSIF), Copyright (c) 2022, authors of DeepSIF,
released under the Clear BSD License. `process_raw_nmm.m`,
`generate_synthetic_source_update.m` and `generate_tvb_hfo.py` are modified versions
of DeepSIF's `process_raw_nmm.m`, `generate_sythetic_source.m` and
`generate_tvb_data.py`; the modifications adapt the pipeline to generate
high-frequency oscillations rather than interictal spikes. Please cite:

> Sun R, Sohrabpour A, Worrell GA, He B. Deep neural networks constrained by neural
> mass models improve electrophysiological source imaging of spatiotemporal brain
> dynamics. *Proc Natl Acad Sci.* 2022;119(31):e2201128119.
> doi:10.1073/pnas.2201128119

Source dynamics are simulated with **The Virtual Brain** (installed separately, not
redistributed here) using the **Jansen–Rit** neural mass model:

> Sanz Leon P, Knock SA, Woodman MM, et al. The Virtual Brain: a simulator of primate
> brain network dynamics. *Front Neuroinformatics.* 2013;7. doi:10.3389/fninf.2013.00010

> Jansen BH, Rit VG. Electroencephalogram and visual evoked potential generation in a
> mathematical model of coupled cortical columns. *Biol Cybern.* 1995;73(4):357-366.
> doi:10.1007/BF00199471

The 998-region connectivity template (`anatomy/connectivity_998.zip`, included in this
repository) is distributed by TVB, not created here.

## Pipeline

Run from the repository root:

```bash
python forward/generate_tvb_hfo.py     # NMM source dynamics (Jansen-Rit, via TVB)
matlab -batch "process_raw_nmm"        # segment raw NMM output
matlab -batch "Select_Source"          # draw cortical source patches
matlab -batch "build_fwd_from_lfd"     # region-wise forward matrix from the lead field
python forward/CovnertToH5.py          # pack to HDF5 for the data loaders
```

`ChangeScaleRatio.m` sets source magnitude to hit a target signal-to-noise ratio;
`mydis.m` and `mynorm.m` are small geometry helpers.
