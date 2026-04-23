#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Qiyuan Feng
Date: Apr 14, 2026 (Copied from MEG repository)

C-SHARP EEG Forward & Inverse Pipeline

Full pipeline description: 
    1. Load data (CTF vs FIF files) - make functions for each data type
    2. Do coregistration, either in line or through YORC script
    3. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
    4. Make evokes for each condition
    5. Create covariance
    6. Forward solution
    7. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
    8. Apply inverse to evokeds
    9. Put into BIDS structure, specifically with events
    10. Generate MNE-report

This module covers steps 5-8 of the full pipeline:
    5. Noise covariance
    6. Forward solution (BEM + source space + leadfield)
    7. Inverse solution (MNE / dSPM / sLORETA / eLORETA / beamformer / custom SVD)
    8. Apply inverse to evokeds

Inputs:
    - evoked : mne.Evoked
    - trans  : mne.Transform or path to -trans.fif

Outputs:
    make_forward  -> fwd : mne.Forward
    make_inverse  -> stc : mne.SourceEstimate, inverse_operator : InverseOperator
"""

# --- Dependencies -----------------------------------------------------------
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
from typing import Tuple, Optional, Literal


# --- Helpers -----------------------------------------------------------------

def _create_mid_surface(subjects_dir, subject, hemisphere):
    """Average white and pial surfaces to produce {hemisphere}.mid."""
    surf_dir = os.path.join(subjects_dir, subject, 'surf')

    white_coords, white_faces = nib.freesurfer.read_geometry(
        os.path.join(surf_dir, f'{hemisphere}.white'))
    pial_coords, pial_faces = nib.freesurfer.read_geometry(
        os.path.join(surf_dir, f'{hemisphere}.pial'))

    if not np.array_equal(white_faces, pial_faces):
        raise ValueError("Face mismatch between white and pial surfaces.")

    mid_coords = (white_coords + pial_coords) / 2.0
    mid_path = os.path.join(surf_dir, f'{hemisphere}.mid')
    nib.freesurfer.write_geometry(mid_path, mid_coords, white_faces)
    print(f"Mid surface saved: {mid_path}")


def _print_source_space_summary(subjects_dir, subject, src):
    """Print vertex counts for white, mid, and decimated source space."""
    surf_dir = os.path.join(subjects_dir, subject, 'surf')

    white_lh, _ = read_surface(os.path.join(surf_dir, 'lh.white'))
    white_rh, _ = read_surface(os.path.join(surf_dir, 'rh.white'))
    mid_lh, _   = read_surface(os.path.join(surf_dir, 'lh.mid'))
    mid_rh, _   = read_surface(os.path.join(surf_dir, 'rh.mid'))

    print(f"  White surface vertices : {white_lh.shape[0] + white_rh.shape[0]}")
    print(f"  Mid surface vertices   : {mid_lh.shape[0] + mid_rh.shape[0]}")
    print(f"  Decimated src vertices : {sum(len(h['vertno']) for h in src)}")


def _visualize_source_space(src, subject, surface):
    """Plot BEM cross-section and 3-D source alignment.

    NOTE: 3-D plots (PyVista/Mayavi) are non-blocking by default in recent
    MNE versions, so they should not stall the pipeline. If running in a
    headless environment set ``mne.viz.set_3d_backend('notebook')`` or skip
    visualization entirely.
    """
    mne.viz.plot_bem(src=src, subject=subject,
                     brain_surfaces=surface, orientation='coronal')
    fig = mne.viz.plot_alignment(
        subject=subject, surfaces="white",
        coord_frame="mri", src=src,
    )
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.30, focalpoint=(-0.03, -0.01, 0.03))


def _visualize_forward(evoked, trans, fwd, subject):
    """Plot sensor-source alignment with forward solution overlay."""
    mne.viz.plot_alignment(
        evoked.info, trans=trans, fwd=fwd,
        subject=subject, surfaces="white",
    )


# --- Covariance Matrix --------------------------------------------------------

# currently, make_inverse is not calling make_cov and takes noise_cov as an input directly
# because this function needs epochs instead of evoked data
def make_cov(epochs, identity=False):
    # simply compute covariance, can be changed later
    noise_cov = mne.compute_covariance(epochs)

    if identity:
        # make the noise covariance identity if:
        # 1) don't have a good estimate of the noise covariance, or
        # 2) the baseline produces unstable results, or
        # 3) for noise-free simulation testing
        n=len(noise_cov.data) 
        for j in range(n):
            for k in range(n):
                if (j==k):
                    noise_cov.data[j][k]=1
                else:
                    noise_cov.data[j][k]=0

def _get_identity_cov(fwd, evoked):
    """Identity noise covariance (for noise-free / simulation testing)."""
    return Covariance(
        data=np.eye(fwd["sol"]["data"].shape[0]),
        names=evoked.info['ch_names'],
        bads=[],
        nfree=1,
        projs=[],
    )


# --- Forward Solution --------------------------------------------------------

def make_forward(subject_id, subjects_dir, trans, evoked,
                 overwrite_fwd=True, overwrite=False,
                 fixed=True, bem_ico=4, src_space="oct7",
                 conductivity=(0.3, 0.006, 0.3),
                 mindist=5, surface='mid',
                 visualize=False, verbose=False, eeg=True, meg=False):
    """Build (or load) the forward solution for a subject.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject name.
    trans : str or mne.Transform
        Head-to-MRI transform (-trans.fif path or Transform object).
    evoked : mne.Evoked
        Used for sensor info when computing the leadfield.
    subjects_dir : str or Path
        FreeSurfer SUBJECTS_DIR.
    overwrite_fwd : bool
        If False, read existing forward if available.
    overwrite : bool
        If True, recompute BEM and source space even if files exist.
    fixed : bool
        Convert to fixed (surface-normal) orientation.
    bem_ico : int
        ICO decimation level for BEM surfaces (higher = finer mesh).
    src_space : str
        Source space decimation, e.g. 'oct6', 'oct7'.
    conductivity : tuple
        BEM conductivities in S/m.
        MEG-only: (0.3,) — single shell.
        EEG+MEG:  (0.3, 0.006, 0.3) — scalp / skull / brain.
    mindist : float
        Exclude sources closer than this (mm) to the inner skull.
    surface : str
        Source surface type ('white', 'pial', or 'mid').
    visualize : bool
        Show BEM / alignment plots (requires a display backend).
    verbose : bool
        Print extra source-space and leadfield diagnostics.

    Returns
    -------
    fwd : mne.Forward
    """
    subject = subject_id
    subjects_dir = str(subjects_dir)
    bem_path = Path(subjects_dir) / subject / 'bem'
    fwd_path = bem_path / f'{subject}-fwd.fif'

    # --- Try loading existing forward ---
    if fwd_path.exists() and not overwrite_fwd:
        print("---- Reading existing forward solution ----")
        fwd = mne.read_forward_solution(str(fwd_path))
        if fixed:
            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
        leadfield = fwd["sol"]["data"]
        print(f"Leadfield: {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
        return fwd

    # --- Build from scratch ---
    print("---- Building forward solution ----")

    # Mid surface generation
    if surface == 'mid':
        for hemi in ['lh', 'rh']:
            _create_mid_surface(subjects_dir, subject, hemi)

    # --- BEM model & solution ---
    if isinstance(conductivity, tuple) and len(conductivity) == 3:
        scale = str(round(1 / (conductivity[1] / conductivity[2])))
    else:
        scale = "1layer"

    bem_fname = bem_path / f'{subject}-scale{scale}-ico{bem_ico}-bem.fif'
    bem_sol_fname = bem_path / f'{subject}-scale{scale}-ico{bem_ico}-bem-sol.fif'

    if not bem_fname.exists() or overwrite:
        mne.bem.make_watershed_bem(subject=subject, overwrite=True,
                                   volume='T1', atlas=True, gcaatlas=True,
                                   show=visualize)
        model = mne.make_bem_model(subject=subject, ico=bem_ico,
                                   conductivity=conductivity)
        mne.write_bem_surfaces(str(bem_fname), model, overwrite=True)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(str(bem_sol_fname), bem, overwrite=True)
    else:
        print("---- Reading existing BEM ----")
        bem = mne.read_bem_solution(str(bem_sol_fname))

    # --- Source space ---
    src_fname = bem_path / f'{subject}-{src_space}-src.fif'

    if not src_fname.exists() or overwrite:
        src = mne.setup_source_space(subject=subject, spacing=src_space,
                                     surface=surface, add_dist=False)
        mne.write_source_spaces(str(src_fname), src, overwrite=True)
        if visualize:
            _visualize_source_space(src, subject, surface)
        if verbose:
            _print_source_space_summary(subjects_dir, subject, src)
    else:
        print("---- Reading existing source space ----")
        src = mne.read_source_spaces(str(src_fname))

    # --- Compute forward ---
    fwd = mne.make_forward_solution(
        evoked.info, trans=trans, src=src, bem=bem,
        eeg=eeg, meg=meg, mindist=mindist, n_jobs=None,
    )
    mne.write_forward_solution(str(fwd_path), fwd, overwrite=True)

    if fixed:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    if verbose:
        leadfield = fwd["sol"]["data"]
        print(f"Source space before fwd: {src}")
        print(f"Source space after fwd:  {fwd['src']}")
        print(f"Leadfield: {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
        print(f"Dipole orientations shape: {fwd['source_nn'].shape}")

    if visualize:
        _visualize_forward(evoked, trans, fwd, subject)

    return fwd



# --- Inverse Solution --------------------------------------------------------

def make_inverse(subjects_dir, subject, fwd, evoked, noise_cov,
                 fixed_ori=True, noise_free=False,
                 snr=8, lambda2=None,
                 inverse_method="MNE",
                 save_dir=None,
                 loose=0.2, depth=0.8):
    """Compute the inverse operator and apply it to an evoked.

    Parameters
    ----------
    subjects_dir : str or Path
    subject : str
    fwd : mne.Forward
    evoked : mne.Evoked
    noise_cov : mne.Covariance
        Ignored when noise_free=True (identity cov is used instead).
    fixed_ori : bool
        True  -> fixed orientation (loose=0, depth=None).
        False -> free/loose orientation using `loose` and `depth` params.
    noise_free : bool
        Use identity covariance and high SNR (simulation mode).
    snr : float
        Signal-to-noise ratio; lambda2 = 1/snr^2 when lambda2 is None.
    lambda2 : float or None
        Explicit regularisation. Overrides snr if provided.
    inverse_method : str
        'MNE', 'dSPM', 'sLORETA', 'eLORETA', 'beamformer', or 'customized'.
    save_dir : str or Path, optional
        If provided, save the resulting STC here.
    loose : float
        Loose orientation constraint (only used when fixed_ori=False).
    depth : float
        Depth weighting exponent (only used when fixed_ori=False).

    Returns
    -------
    stc : mne.SourceEstimate
    inverse_operator : InverseOperator
    """
    print("---- Computing inverse ----")

    # Noise covariance
    if noise_free:
        noise_cov = _get_identity_cov(fwd, evoked)
        snr = 100
    # noise_cov = noise_cov.pick_channels(evoked.ch_names)
    common_chs = [ch for ch in evoked.ch_names if ch in noise_cov.ch_names]
    noise_cov = noise_cov.pick_channels(common_chs)

    # Regularisation
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2
    print(f"  SNR={snr}, lambda2={lambda2:.4f}")

    # Inverse operator
    if fixed_ori:
        inverse_operator = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=0, depth=None, fixed=True)
    else:
        inverse_operator = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=loose, depth=depth, fixed=False)

    inv_fname = Path(subjects_dir) / subject / 'bem' / f'{subject}-inv.fif'
    write_inverse_operator(str(inv_fname), inverse_operator, overwrite=True)

    # Apply inverse
    if inverse_method == "beamformer":
        filters = make_lcmv(
            info=evoked.info, forward=fwd, data_cov=noise_cov,
            reg=0.05,
            pick_ori=None if fixed_ori else 'normal',
            rank='info', verbose=False,
        )
        stc = apply_lcmv(evoked, filters)

    elif inverse_method == "customized":
        stc = _pseudo_inverse_custom(subject, fwd, evoked, snr,
                                     fixed_ori=fixed_ori)
    else:
        # MNE, dSPM, sLORETA, eLORETA
        stc = apply_inverse(
            evoked, inverse_operator, lambda2,
            method=inverse_method, pick_ori=None, verbose=True,
        )

    # Save STC if output directory was given
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stc_fname = str(save_dir / f'{subject}-{inverse_method}')
        stc.save(stc_fname, overwrite=True)  # mne.SourceEstimate.save()
        print(f"  STC saved: {stc_fname}")

    return stc, inverse_operator

def get_filtered_stc(fwd, epochs, cov,
                     filters=((5.0, "even", "nF_center"),
                              (6.0, "even", "nF_upper"),
                              (7.5, "even", "nF_lower")),
                     max_harmonic_order=4,
                     fixed_ori=True,
                     snr=8.0, lambda2=None,
                     inverse_method="MNE",
                     save_dir=None):
    """Frequency-domain filtering pipeline for SSVEP source localization.

    For each target stimulus frequency, this function:
      1. FFTs every epoch individually (full two-sided spectrum).
      2. Zeros out all bins except the requested harmonics of the target freq.
      3. iFFTs each filtered epoch back to the time domain.
      4. Averages the filtered time series across epochs: get filtered evoked.
      5. Applies the inverse operator to get a source estimate (STC).

    Parameters
    ----------
    fwd : mne.Forward
    epochs : mne.Epochs
        Core stimulus epochs (e.g. epochs['bin']).
    cov : mne.Covariance
        Noise covariance.
    filters : sequence of (base_freq, harmonics, label)
        Each entry defines one source localization run:
        - base_freq  : float — stimulus base frequency in Hz (e.g. 6.0)
        - harmonics  : "even" | "all" | "odd" — which harmonic orders to keep
        - label      : str   — key used in the returned dict (e.g. "nF_upper")
    max_harmonic_order : int
        Stop including harmonics above this order (e.g. 4 → up to 4*f0).
    fixed_ori : bool
    snr : float
    lambda2 : float or None
    inverse_method : str
    save_dir : str or Path, optional
    """
    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
    eeg_info  = mne.pick_info(epochs.info, eeg_picks)

    # average_epochs=False: we need per-epoch complex spectra so we can filter
    # each epoch and iFFT back before averaging the time series.
    freqs, spectra = _time_to_freq(epochs, full=True, detrend=False,
                                   unit="raw", average_epochs=False)
    # spectra shape: (n_epochs, n_freqs, n_channels)

    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2

    filtered_stcs = {}
    for base_freq, harmonics, label in filters:
        # Filter spectrum for this target frequency 
        fft_filtered = _filter_freq_data(freqs, spectra,
                                         base_freqs=base_freq,
                                         harmonics=harmonics,
                                         include_dc=False,
                                         max_order=max_harmonic_order)
        # fft_filtered shape: (n_epochs, n_freqs, n_channels)

        # iFFT each epoch, average filtered time series across epochs
        ts_epochs = []
        for ep_fft in fft_filtered:                  # ep_fft: (n_freqs, n_channels)
            ts = _freq_to_time(ep_fft, n_times=len(epochs.times), full=True,
                               unit="raw", add_mean=None, n_fft=None)
            ts_epochs.append(ts)                     # ts: (n_times, n_channels)
        ts_mean = np.mean(ts_epochs, axis=0)         # (n_times, n_channels)

        # Build evoked from filtered mean time series
        evoked = mne.EvokedArray(
            ts_mean.T,                               # (n_channels, n_times)
            eeg_info,
            tmin=float(epochs.times[0]),
            nave=len(epochs),
            comment=f"SSVEP filtered {base_freq} Hz ({harmonics} harmonics)",
        )

        if save_dir is not None:
            fig = evoked.plot_joint(picks="eeg", show=False)
            fig.savefig(f"{save_dir}/eeg_evoked_{label}.png",
                        dpi=300, bbox_inches="tight", pad_inches=0)

        # ── Inverse solution ──────────────────────────────────────────────────
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            evoked.info, fwd, cov,
            loose=0 if fixed_ori else 0.2,
            depth=None if fixed_ori else 0.8,
            fixed=fixed_ori,
        )
        stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator, lambda2,
            method=inverse_method, pick_ori=None, verbose=False,
        )
        filtered_stcs[label] = {
            "inv_op": inverse_operator,
            "stc":    stc,
        }
        print(f"  [{label}] {base_freq} Hz {harmonics} harmonics: STC computed")

    return filtered_stcs

def _time_to_freq(
    wave,                               # np.ndarray (n_times, n_channels) OR mne.Epochs
    sfreq: float = 360.0,               # ignored when wave is mne.Epochs (taken from info)
    n_fft: Optional[int] = None,        # FFT length. If None, uses n_times. Can be increased for zero-padding.
    detrend: bool = True,               # If True, remove the mean (DC) from each channel before FFT.
    full: bool = False,                 # False: single-sided rFFT; True: full complex FFT.
    unit: Literal["raw", "real"] = "raw", # "raw": unnormalized FFT coefficients; "real": normalized, np.abs(fft_c)=real-valued amplitude, (accounts for one-sided folding and 1/N scaling).
    average_epochs: bool = True,        # mne.Epochs only: if True average amplitude spectra across epochs;
                                        # if False return per-epoch spectra (n_epochs, n_freqs, n_channels).
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the FFT of time-domain EEG data.
    Accepts either a raw numpy array or an mne.Epochs

    Returns: 
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    fft_c : ndarray, complex
        Fourier coefficients.  Shape (n_freqs, n_channels) when input is a
        numpy array or average_epochs=True; (n_epochs, n_freqs, n_channels)
        when average_epochs=False.
        - amplitude : np.abs(fft_c)
        - phase     : np.angle(fft_c)
        - power     : np.abs(fft_c) ** 2
    """
    # ── Handle mne.Epochs input ───────────────────────────────────────────────
    if isinstance(wave, mne.BaseEpochs):
        sfreq = wave.info['sfreq']
        # get_data returns (n_epochs, n_channels, n_times); transpose each epoch
        # to (n_times, n_channels) to match the array convention below.
        data = wave.get_data(picks='eeg')          # (n_epochs, n_ch, n_times)
        epoch_spectra = []
        for epoch in data:                          # epoch: (n_ch, n_times)
            X = epoch.T                             # → (n_times, n_ch)
            _, fft_ep = _time_to_freq(X, sfreq=sfreq, n_fft=n_fft,
                                      detrend=detrend, full=full, unit=unit)
            epoch_spectra.append(fft_ep)

        # epoch_spectra: list of (n_freqs, n_ch) arrays
        stacked = np.array(epoch_spectra)           # (n_epochs, n_freqs, n_ch)

        # Recompute freq vector from one call (same for every epoch)
        n_times_ep = data.shape[-1]
        n_fft_ep   = n_times_ep if n_fft is None else max(n_fft, n_times_ep)
        if full:
            freqs = np.fft.fftfreq(n_fft_ep, d=1.0 / sfreq)
        else:
            freqs = np.fft.rfftfreq(n_fft_ep, d=1.0 / sfreq)

        if average_epochs:
            # Average complex spectra first, THEN take amplitude if needed.
            # Averaging magnitudes (np.abs first) would prevent noise cancellation.
            return freqs, np.mean(stacked, axis=0)   # complex, (n_freqs, n_ch)
        else:
            return freqs, stacked                    # complex, (n_epochs, n_freqs, n_ch)

    # ── Original array path ───────────────────────────────────────────────────
    n_times, n_ch = wave.shape

    if n_fft is None:
        n_fft = n_times
    if n_fft < n_times:
        raise ValueError("n_fft must be >= n_times (zero-padding allowed, truncation not supported).")

    X = wave.astype(float, copy=True)

    if detrend:
        X -= X.mean(axis=0, keepdims=True)

    if full:
        fft_c = np.fft.fft(X, n=n_fft, axis=0)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / sfreq)
    else:
        fft_c = np.fft.rfft(X, n=n_fft, axis=0)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

    if unit == "real":
        fft_c = fft_c / n_times
        if not full:
            if n_fft % 2 == 0 and fft_c.shape[0] >= 2:
                fft_c[1:-1, :] *= 2.0   # even: double all except DC and Nyquist
            else:
                fft_c[1:, :] *= 2.0     # odd: double all except DC

    return freqs, fft_c

def _freq_to_time(
    fft_c: np.ndarray,                # spectrum from time_to_freq
    n_times: int,                     # original time length — must be provided explicitly
    full: bool = False,               # same value you used in time_to_freq
    unit: Literal["raw", "real"] = "raw",
    add_mean: Optional[np.ndarray] = None,  # per-channel means to add back
    n_fft: Optional[int] = None,    # n_fft used in forward; if None -> n_times
) -> np.ndarray:

    F = fft_c.copy()
    if n_fft is None:
        n_fft = n_times

    # Undo amplitude normalization if needed
    if unit == "real":
        if not full:
            # rFFT case: we previously doubled non-DC/non-Nyquist bins
            if (n_fft % 2 == 0) and (F.shape[0] >= 2):
                # even n_fft -> last bin is Nyquist (do NOT halve DC or Nyquist)
                F[1:-1, :] *= 0.5
            else:
                # odd n_fft -> no Nyquist; halve everything except DC
                F[1:, :] *= 0.5
        # undo the 1/N scaling applied in forward
        F *= n_times

    # Inverse transform
    if full:
        wave_rec = np.fft.ifft(F, n=n_times, axis=0)
        # A full FFT of a real signal must have conjugate symmetry.
        # If the imaginary residual is large, the spectrum was not symmetric
        # (e.g. only one side of a ±f pair was kept), and .real is silently wrong.
        imag_max = np.max(np.abs(wave_rec.imag))
        real_max = np.max(np.abs(wave_rec.real)) + 1e-30
        if imag_max / real_max > 1e-6:
            raise ValueError(
                f"ifft produced significant imaginary residual "
                f"(imag/real = {imag_max/real_max:.2e}). "
                "The spectrum likely lacks conjugate symmetry — check that "
                "_filter_freq_data kept both +f and -f bins."
            )
        wave_rec = wave_rec.real
    else:
        wave_rec = np.fft.irfft(F, n=n_times, axis=0)

    # Optionally add back channel means
    if add_mean is not None:
        add_mean = np.asarray(add_mean)
        if add_mean.ndim != 1 or add_mean.shape[0] != wave_rec.shape[1]:
            raise ValueError("add_mean must have shape (n_channels,)")
        wave_rec = wave_rec + add_mean.reshape(1, -1)

    return wave_rec

def _filter_freq_data(freqs, fft_c, base_freqs, harmonics="even",
                      include_dc=False, max_order=None):
    """Zero out all spectrum bins except those matching requested harmonics.

    Parameters
    ----------
    freqs : (n_freqs,) array
        Frequency vector from np.fft.fftfreq (full two-sided FFT assumed).
    fft_c : complex array, shape (n_freqs, n_channels) OR (n_epochs, n_freqs, n_channels)
        Complex full-spectrum coefficients from _time_to_freq with full=True.
    base_freqs : float or list of float
        Stimulus base frequency/frequencies in Hz, e.g. 6.0 or [5.0, 6.0, 7.5].
    harmonics : {"even", "all", "odd"}
        Which harmonic orders to keep:
        - "even" (default): 2*f0, 4*f0, 6*f0, … (SSVEP standard — even harmonics
          only, since odd harmonics are typically suppressed in SSVEP responses)
        - "all" : f0, 2*f0, 3*f0, … (all harmonics)
        - "odd" : f0, 3*f0, 5*f0, … (odd harmonics only)
    include_dc : bool
        Keep DC (0 Hz). Default False.
    max_order : int or None
        Stop at this harmonic order. None = keep all within Nyquist.

    Returns
    -------
    fft_filt : complex array, same shape as fft_c
        Filtered spectrum with non-kept bins zeroed out.
    """
    if np.isscalar(base_freqs):
        base_freqs = [float(base_freqs)]
    else:
        base_freqs = [float(f) for f in base_freqs]

    if harmonics == "even":
        keep_even, keep_odd = True, False
    elif harmonics == "all":
        keep_even, keep_odd = True, True
    elif harmonics == "odd":
        keep_even, keep_odd = False, True
    else:
        raise ValueError(f"harmonics must be 'even', 'all', or 'odd'; got {harmonics!r}")

    # Frequency resolution from the smallest positive step in the freq vector
    pos   = np.sort(np.unique(np.abs(freqs)))
    diffs = np.diff(pos)
    df    = float(diffs[diffs > 0][0]) if np.any(diffs > 0) else 0.0
    f_max = float(np.max(np.abs(freqs))) if freqs.size else 0.0

    keep = np.zeros(len(freqs), dtype=bool)

    if include_dc:
        keep |= (freqs == 0.0)

    def mark_exact(f_target: float):
        k     = int(round(f_target / df)) if df > 0 else 0
        f_bin = k * df
        if not np.isclose(f_bin, f_target, atol=1e-9):
            raise ValueError(
                f"Target {f_target} Hz is not on the FFT grid (df={df:.6f} Hz). "
                "Adjust epoch length so that the stimulus frequency divides evenly."
            )
        if f_bin > 0:
            # Use isclose rather than == : exact float equality can silently miss
            # the bin if f_bin and freqs[k] differ by even one ULP, zeroing the
            # entire harmonic and producing a silent all-zero STC.
            keep[:] |= np.isclose(np.abs(freqs), f_bin, atol=df * 0.01)

    for f0 in base_freqs:
        if f0 <= 0:
            continue
        k = 1
        while True:
            if max_order is not None and k > max_order:
                break
            f_target = k * f0
            if f_target > f_max:
                break
            is_even = (k % 2 == 0)
            if (is_even and keep_even) or ((not is_even) and keep_odd):
                mark_exact(f_target)
            k += 1

    # Apply mask — handle both (n_freqs, n_ch) and (n_epochs, n_freqs, n_ch)
    fft_filt = np.zeros_like(fft_c)
    if fft_c.ndim == 2:
        fft_filt[keep, :] = fft_c[keep, :]
    elif fft_c.ndim == 3:
        fft_filt[:, keep, :] = fft_c[:, keep, :]
    else:
        raise ValueError(f"fft_c must be 2-D or 3-D, got shape {fft_c.shape}")
    return fft_filt

# --- Custom SVD pseudo-inverse ----------------------------------------------

def _pseudo_inverse_custom(subject, fwd, evoked, snr, fixed_ori):
    """Truncated SVD pseudo-inverse (for experimentation / comparison).

    Computes A_pinv via SVD of the leadfield, thresholding small singular
    values based on the SNR.  For free orientation, the 3-component source
    vector is collapsed to its norm.
    """
    A = fwd["sol"]["data"]  # sensors x dipoles
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    threshold = np.max(Sigma) / (snr + 1)
    Sigma_inv = np.array([1/s if s > threshold else 0 for s in Sigma])
    n_kept = int(np.sum(Sigma > threshold))

    print("---- Custom pseudo-inverse (SVD) ----")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Kept {n_kept}/{len(Sigma)} singular values")

    A_pinv = Vt.T @ np.diag(Sigma_inv) @ U.T
    X = A_pinv @ evoked.data  # estimated source activity

    # Collapse xyz triplets to vector norm for free orientation
    if not fixed_ori:
        n_sources = X.shape[0] // 3
        X = np.linalg.norm(X.reshape(n_sources, 3, -1), axis=1)

    print(f"  Source estimate shape: {X.shape}")

    vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc = mne.SourceEstimate(
        data=X, vertices=vertices,
        tmin=evoked.times[0],
        tstep=1.0 / evoked.info['sfreq'],
        subject=subject,
    )
    return stc


# --- Main (example usage) ---------------------------------------------------

if __name__ == '__main__':
    subjects_dir = Path(os.environ["SUBJECTS_DIR"])
    # Load trans and evoked, then:
    #   fwd = make_forward(subject_id, trans, evoked, subjects_dir=subjects_dir)
    #   stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
