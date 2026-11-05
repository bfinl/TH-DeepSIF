# model_weights/

`model_weights.pt` (126.6 MB) holds the trained TH-DeepSIF weights used for every result
in the paper. It exceeds GitHub's 100 MB per-file limit, so it is attached to the
repository's latest [Release](../../releases) rather than committed here.

```bash
# from the repository root
curl -L -o model_weights/model_weights.pt \
  https://github.com/bfinl/TH-DeepSIF/releases/latest/download/model_weights.pt
```

The evaluation scripts expect it at `model_weights/44_the_model/model_best.pth.tar`:

```bash
mkdir -p model_weights/44_the_model
mv model_weights/model_weights.pt model_weights/44_the_model/model_best.pth.tar
```

The file is the training checkpoint with the Adam optimizer state removed (380 MB to
126.6 MB); weights are unchanged fp32. It contains `state_dict`, `arch`, `epoch`, `lr`
and the training and test set names.
