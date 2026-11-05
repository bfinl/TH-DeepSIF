# TH-DeepSIF: Transformer-based Deep Learning Source Imaging Framework for High-Frequency Oscillations

Code for:

> Rong J, Cai Z, Gonsisko C, Joseph B, Worrell GA, He B.
> **Transformer-based EEG Source Imaging Localizes Pathological High-Frequency Oscillations in Focal Epilepsy.**
> *Brain Communications*, 2026. doi: [10.1093/braincomms/fcag431](https://doi.org/10.1093/braincomms/fcag431)

TH-DeepSIF maps multi-channel scalp EEG onto 994 cortical regions. It is trained end-to-end
on neural mass model (NMM) simulations of high-frequency oscillations propagated through a
boundary element head model, and is applied without patient-specific parameter tuning.

It extends [DeepSIF](https://github.com/bfinl/DeepSIF)
([Sun et al., *PNAS* 2022](https://doi.org/10.1073/pnas.2201128119)) by replacing the
recurrent temporal model with a transformer encoder, which is what makes brief,
low-amplitude HFO events tractable.

All scripts are run from the repository root.

## Prior work and credits

The contribution specific to this work is the transformer architecture (`network.py`),
its training and evaluation for high-frequency oscillations, and the clinical validation
and analysis code in `misc_scripts/`.

**The neural mass model simulation and forward modelling pipeline in `forward/` is not
original to this project.** It is derived from the DeepSIF codebase
(https://github.com/bfinl/DeepSIF), Copyright (c) 2022, authors of DeepSIF, released
under the Clear BSD License; `process_raw_nmm.m`, `generate_synthetic_source_update.m`
and `generate_tvb_hfo.py` are modified versions of files from that repository, adapted to
simulate high-frequency oscillations rather than interictal spikes. Each file records
this in its header, and `forward/README.md` gives the details.

Source dynamics are simulated with [The Virtual Brain](https://www.thevirtualbrain.org/)
(installed separately, not redistributed here) using the Jansen–Rit neural mass model.
Please cite the following alongside this work:

> Sun R, Sohrabpour A, Worrell GA, He B. Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics. *Proc Natl Acad Sci.* 2022;119(31):e2201128119. doi:10.1073/pnas.2201128119
>
> Sanz Leon P, Knock SA, Woodman MM, et al. The Virtual Brain: a simulator of primate brain network dynamics. *Front Neuroinformatics.* 2013;7. doi:10.3389/fninf.2013.00010
>
> Jansen BH, Rit VG. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biol Cybern.* 1995;73(4):357-366. doi:10.1007/BF00199471

See `NOTICE` for the retained copyright notices.

## Train Data Generation

NMM source activity is simulated in The Virtual Brain, processed in MATLAB, and packed
into HDF5 for the data loaders:

```bash
python forward/generate_tvb_hfo.py     # NMM HFO source dynamics
matlab -batch "process_raw_nmm"        # assemble raw NMM output
matlab -batch "Select_Source"          # draw source patches
matlab -batch "build_fwd_from_lfd"     # region-wise forward matrix
python forward/CovnertToH5.py          # pack to HDF5
```

## Training

The configuration of the published model (`model_id 44`):

```bash
python main.py \
    --train train_998_snr15_extenE7.mat \
    --test  test_998_snr15_extenL5.mat \
    --arch EnhancedEEGToBrainRegionTransformer \
    --model_id 44 \
    --fwd leadfield_75_20k.mat \
    --dat HFOEEGBuildJR \
    --num_layer 6 --drop_out 0.3 --lr 3e-4 --epoch 50 --batch_size 64 \
    --inv_lfd 1 --is_highpassed 1 --highpassed_eeg 80 \
    --prob_add_spike 0.5 --variable_length 1 --workers 12
```

The best checkpoint is written to `model_weights/44_the_model/model_best.pth.tar`.

## Evaluation

Simulated data, single-source then two-source (Figure 3, Supplementary Figure 2):

```bash
python eval_sim.py --test test_998_snr15_extE4S1.mat \
    --arch EnhancedEEGToBrainRegionTransformer --model_id 44 --num_layer 6 \
    --fwd leadfield_75_20k.mat --dat HFOEEGBuildEvalJR \
    --inv_lfd 1 --is_hfo 1 --prob_add_spike 0.5 --highpassed_eeg 80 --workers 5

python eval_sim.py --test test_998_snr15_extenE4.mat \
    --arch EnhancedEEGToBrainRegionTransformer --model_id 44 --num_layer 6 \
    --fwd leadfield_75_20k.mat --dat HFOEEGBuildEvalJR \
    --inv_lfd 1 --is_hfo 1 --prob_add_spike 0.5 --highpassed_eeg 80 --workers 5
```

### Quick check: the bundled example patient

One de-identified pathological HFO event is included so the clinical pipeline can be
run without access to the full cohort. With the trained weights in place:

```bash
python eval_real.py \
    --patient_id 1 --patient_root example_data \
    --model_id 44 --arch EnhancedEEGToBrainRegionTransformer --num_layer 6 \
    --num_electrode 75 --spike_type p \
    --highpass_eeg 80 --suppress_distant_hfo 1 \
    --output_folder example_out

python example_data/evaluate_example.py \
    --pred_dir model_weights/44_the_model/example_out
```

This should report a localization error of about 2.2 mm against the resection.
See `example_data/README.md` for the expected output and figures showing the
reconstructed source on the cortex and its waveform inside and outside the
resection.

### Full cohort

Clinical scalp EEG (Figures 4-5, Supplementary Figures 3-6):

```bash
python eval_real.py \
    --model_id 44 --arch EnhancedEEGToBrainRegionTransformer --num_layer 6 \
    --patient_id "<comma-separated patient ids>" \
    --num_electrode 75 --spike_type p --patient_dataset pHFO \
    --highpass_eeg 80 --suppress_distant_hfo 1 --output_folder test80200
```

### Statistics

`misc_scripts/statistical_tests.py` contains every inferential test in the paper, with
the figure each one produced noted in the docstrings: paired Wilcoxon with Holm
correction for the simulated comparisons (Figure 3), and linear mixed-effects models
with a patient random intercept and split Holm correction for the clinical
comparisons (Figures 4-5, Supplementary Figures 4-6). `snr_stratified_stats.py` and
`supp_lme_stats.py` are the runnable scripts for Supplementary Figures 2, 4 and 6.

`misc_scripts/metrics.py` defines the six localization metrics computed against the
resection — localization error, sensitivity, precision, specificity, F1 and geometric
mean — including the balanced near/far negative sampling used for specificity.
`example_data/evaluate_example.py` reports all six for the bundled event.

`misc_scripts/` also holds the sLORETA benchmark (`sloreta_sim.py`, `sloreta_real.py`,
both MNE-Python). Plotting code is not included.

## Data

The head model, lead field, cortical geometry and noise templates in `anatomy/` are included
in this repository. The trained weights (126.6 MB) exceed GitHub's 100 MB per-file limit and are
attached to the latest [Release](../../releases):

```bash
mkdir -p model_weights/44_the_model
curl -L -o model_weights/44_the_model/model_best.pth.tar \
  https://github.com/bfinl/TH-DeepSIF/releases/latest/download/model_weights.pt
```

The simulated evaluation sets behind Figure 3 and Supplementary Figure 2 (1,000
single-source and 1,000 two-source trials, with a loader script) are on Figshare:
https://doi.org/10.6084/m9.figshare.33496762. The full simulated training set (a 120 GB
waveform bank) is not distributed; regenerate it with `forward/` as described above.

A de-identified example event is included in `example_data/` so the clinical
pipeline can be run end to end; the full cohort is not distributed.

## Dependencies

```bash
pip install -r requirements.txt
```

Python 3.7 with PyTorch 1.4 for training and evaluation; the statistics scripts in
`misc_scripts/` additionally need `statsmodels`. The NMM simulation in `forward/`
needs The Virtual Brain (`tvb-library`) and `PyWavelets`, and MATLAB R2020b or later for
its `.m` scripts.

## Citation

Please cite the paper above, if using the code. If you use the simulation pipeline in
`forward/`, please also cite Sun et al., *PNAS* 2022, The Virtual Brain, and Jansen &
Rit — see **Prior work and credits** above.

## License

BSD-3-Clause-Clear. See LICENSE.txt, and `NOTICE` for retained third-party copyright
notices.
