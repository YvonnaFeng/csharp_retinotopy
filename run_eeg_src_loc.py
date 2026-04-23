import os
import warnings
import numpy as np
import nibabel as nib
import mne
from mne import Covariance
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator,
                               apply_inverse)
from mne.beamformer import make_lcmv, apply_lcmv
from mne.surface import read_surface
from pathlib import Path
# This takes care some numpy dependency issues...not required depending on the numpy version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from make_forward_inverse import make_forward, make_inverse, get_filtered_stc


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

filtered_stcs = get_filtered_stc(fwd, epochs_core, cov,
                     filters=((5.0, "even", "center"),
                              (6.0, "even", "upper"),
                              (7.5, "even", "lower")),
                     max_harmonic_order=4,
                     fixed_ori=True,
                     snr=3, lambda2=None,
                     inverse_method=inverse_method,
                     save_dir=None)

# --- Visualize inverse --------------------------------------------
for label, result in filtered_stcs.items():
    stc = result["stc"]
    vertno_max, time_max = stc.get_peak(hemi="rh")

    surfer_kwargs = dict(
        hemi="split",
        subjects_dir=subjects_dir,
        views=["caudal", "medial"],
        initial_time=time_max,
        time_unit="s",
        size=(800, 800),
        smoothing_steps=5)

    brain = stc.plot(**surfer_kwargs)

    # Colorbar layout (tuned for split hemi, two views)
    brain.plotter.scalar_bar.GetLabelTextProperty().SetFontSize(8)
    sb = brain.plotter.scalar_bar
    x, _ = sb.GetPosition()
    sb.SetPosition(x, 0.6)
    w, h = sb.GetPosition2()
    sb.SetPosition2(w, h * 0.5)
    for renderer in brain.plotter.renderers:
        for actor in renderer.GetActors2D():
            if hasattr(actor, 'GetTextProperty'):
                actor.GetTextProperty().SetFontSize(7)

    brain.add_foci(
        vertno_max,
        coords_as_verts=True,
        hemi="rh",
        color="blue",
        scale_factor=0.6,
        alpha=0.5)
    brain.add_text(0.1, 0.9, f"{inverse_method}: {label}", "title", font_size=8)

    video_path = os.path.join(save_dir, f"{inverse_method}_{label}.mp4")
    brain.save_movie(video_path, time_dilation=2, framerate=24)
    print(f"Saved video → {video_path}")
    brain.close()

