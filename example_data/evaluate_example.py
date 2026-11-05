"""Score the example patient's reconstructions against the resection.

Run after eval_real.py has written its per-event .mat files:

    python example_data/evaluate_example.py --pred_dir model_weights/44_the_model/example_out

Reports, per event, the localization error between the reconstructed source
region and the surgical resection, using the same Otsu thresholding and
region-level distance metric as the paper. Pure Python -- no MATLAB needed.
"""
import argparse
import glob
import re
import os
import sys
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


sys.path.insert(0, os.path.join(ROOT, "misc_scripts"))
from metrics import all_metrics   # noqa: E402  (the paper's metric definitions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True,
                    help="folder of *_spike*.mat files written by eval_real.py")
    ap.add_argument("--resection",
                    default=os.path.join(HERE, "patient1",
                                         "projected_resection_to_fs_cortex.mat"))
    ap.add_argument("--dis_matrix",
                    default=os.path.join(ROOT, "anatomy", "dis_matrix_fs_20k.mat"))
    args = ap.parse_args()

    resection = loadmat(args.resection)["resection_region"].ravel().astype(int)
    dis = loadmat(args.dis_matrix)["raw_dis_matrix"]

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*_spike*.mat")))
    if not files:
        raise SystemExit("no prediction files found in %s -- run eval_real.py first"
                         % args.pred_dir)

    print("resection covers %d of 994 cortical regions" % resection.size)
    print()
    print("%-6s %-9s %-8s %-6s %-6s %-6s %-6s %-6s"
          % ("event", "n_active", "LE(mm)", "sens", "prec", "spec", "F1", "GM"))
    errors = []
    rng = np.random.default_rng(0)                 # seeds the near-negative sample
    for f in files:
        recon = np.squeeze(loadmat(f)["reconstruction"])
        mid = recon.shape[0] // 2
        half = int(round(0.050 * 500 / 2))
        power = np.max(np.abs(recon[mid - half:mid + half + 1]), axis=0)
        m = all_metrics(power, resection, dis, rng=rng)
        ev = re.search(r"spike(\d+)", os.path.basename(f)).group(1)
        if m is None:
            print("%-6s %-9s %s" % (ev, 0, "n/a")); continue
        errors.append(m["le"])
        print("%-6s %-9d %-8.2f %-6.3f %-6.3f %-6.3f %-6.3f %-6.3f"
              % (ev, m["n_active"], m["le"], m["re"], m["pr"], m["sp"], m["f1"], m["gm"]))

    if errors:
        print()
        if len(errors) == 1:
            print("localization error: %.2f mm" % errors[0])
        else:
            print("median localization error: %.2f mm  (n = %d events)"
                  % (np.median(errors), len(errors)))
        print()
        print("Expected: about 2.2 mm for the bundled event. For reference, the")
        print("published cohort median was 10.9 mm across 70 events in 20 patients.")


if __name__ == "__main__":
    main()
