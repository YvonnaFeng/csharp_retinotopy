import os
import warnings
import numpy as np
import mne
from pathlib import Path
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator,
                               apply_inverse)
from mne.beamformer import make_lcmv, apply_lcmv
# This takes care some numpy dependency issues...not required depending on the numpy version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from make_forward_inverse import make_forward, make_inverse, get_filtered_stc
from viz_src import viz_filtered_stcs

from config import sample_dir, raw_files, epoch_files, trans, subjects_dir, subject, viz_bool, save_report, save_dir

for file in epoch_files:
    if file.endswith(".fif"):
        # read in epochs
        epochs = epochs = mne.read_epochs(os.path.join(sample_dir,file), preload=True)

# After rejection — split by condition
epochs_core  = epochs['bin']            # 'bin/0'–'bin/4'
epochs_noise = epochs['noise']          # prelude + postlude combined

evoked = epochs_core.average() # only average the core 5 bins

# create noise covariance
# cov = mne.compute_covariance(epochs_noise, tmin=None, tmax=None, method='empirical')
# cov = mne.cov.regularize(cov, evoked.info, eeg=0.1, exclude = 'bads')
# mne.viz.plot_cov(cov, epochs.info)
# cov.save(f'{sample_dir}/empirical_reg_cov.fif')
cov = mne.read_cov(f'{sample_dir}/empirical_reg_cov.fif')

# forward solution
fwd = make_forward(subject, subjects_dir, trans, evoked,
                 overwrite_fwd=False, overwrite=False,
                 fixed=True, bem_ico=4, src_space="oct7",
                 conductivity=(0.3, 0.006, 0.3),
                 mindist=5, surface='mid',
                 visualize=False, verbose=False)

# Inverse
inverse_method = "dSPM"
# stc, inv_op = make_inverse(subjects_dir, subject, fwd, evoked, cov,
#                  fixed_ori=True, noise_free=False,
#                  snr=3, lambda2=None,
#                  inverse_method=inverse_method,
#                  save_dir=save_dir,
#                  loose=0.2, depth=0.8)

filters_def = (
    (5.0, "even", "center"),
    (6.0, "even", "upper"),
    (7.5, "even", "lower"),
)
freq_map      = {lbl: f0 for f0, _, lbl in filters_def}
harmonics_map = {lbl: h  for _, h,  lbl in filters_def}

filtered_stcs = get_filtered_stc(fwd, epochs_core, cov,
                     filters=filters_def,
                     max_harmonic_order=4,
                     fixed_ori=True,
                     snr=3, lambda2=None,
                     inverse_method=inverse_method,
                     save_dir=None)

# --- Load FreeSurfer visual area labels (V1/V2/V3, rh only) ------------------
lbl_dir = Path(subjects_dir) / subject / "label"
visual_area_specs = [
    ("rh.V1_exvivo.thresh.label", "black"),
    # ("rh.V2_exvivo.thresh.label", "red"),
    # ("rh.V3_exvivo.thresh.label", "green"),
]
visual_labels = []
for lbl_fname, color in visual_area_specs:
    lbl_path = lbl_dir / lbl_fname
    if lbl_path.exists():
        try:
            vl = mne.read_label(str(lbl_path), subject=subject)
            visual_labels.append((vl, color))
            print(f"Loaded label: {lbl_fname}")
        except Exception as e:
            print(f"Could not load {lbl_fname}: {e}")
    else:
        print(f"Label not found (skipping): {lbl_path}")

# --- Visualize ---------------------------------------------------------------
viz_filtered_stcs(
    filtered_stcs,
    freq_map,
    harmonics_map,
    subjects_dir=subjects_dir,
    inverse_method=inverse_method,
    visual_labels=visual_labels,
    mode="snapshot",          # "snapshot" | "video"
    n_snapshots=5,
    views=("caudal", "medial"),
    show_evoked=None,         # None = auto (True for snapshot, False for video)
    occ_channels=None,        # None = auto-detect occipital channels
    save_dir=save_dir,        # None = interactive window
    marker="com",             # "peak" | "com" | None
    com_top_num=20,           # top-N rh vertices used for COM centroid
)

