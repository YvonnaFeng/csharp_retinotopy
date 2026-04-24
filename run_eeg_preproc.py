import mne
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so figures don't pop up during report building
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from autoreject import AutoReject
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

subject_id    = "S001"
recording_str = "0403_125105"
data_path     = "~/csharp_data"

# ── Step toggles ──────────────────────────────────────────────────────────────
DO_FILTERING          = True
DO_BAD_CHANNELS       = True
DO_ICA                = False   # set True to run ICA
DO_REREFERENCING      = True
DO_EPOCHING           = True
DO_ARTIFACT_REJECTION = True
DO_SSVEP_PSD          = True
DO_SSVEP_SNR          = True

# ── Output toggles ────────────────────────────────────────────────────────────
SAVE_REPORT  = True   # write the HTML report to disk
SHOW_FIGURES = False  # pop up interactive windows (requires an interactive backend)

# ── Filter cutoffs ────────────────────────────────────────────────────────────
FILTER_HIGH = 0.5   # high-pass cutoff (Hz) → l_freq; removes slow drifts
FILTER_LOW  = 85    # low-pass  cutoff (Hz) → h_freq; removes high-freq noise

# ── Bad channels (set before running) ─────────────────────────────────────────
BAD_CHANNELS = ['E17']
INTERPOLATE_BADS = True

# ── ICA settings ──────────────────────────────────────────────────────────────
ICA_N_COMPONENTS  = 25
ICA_EXCLUDE_COMPS = [13]     # component indices to remove
ICA_EOG_PROXY_CH  = 'E25'   # frontal channel used as EOG proxy

# ── Epoching ──────────────────────────────────────────────────────────────────
EPOCH_TMIN = 0.0
EPOCH_TMAX = 2.0
# core_only=False (default): include prelude + postlude labeled as
# 'noise/prelude' and 'noise/postlude' alongside core bins 'bin/0'–'bin/4'.
# core_only=True: only the 5 core bins (original behaviour).
EPOCH_CORE_ONLY = False

# ── Artifact rejection ────────────────────────────────────────────────────────
# 'autoreject'  — recommended default: data-driven per-channel thresholds,
#                 interpolates bad channel-epochs before dropping whole epochs
# 'peak_to_peak' — simple fixed-threshold drop (fast but arbitrary, no interpolation)
# 'both'         — peak-to-peak coarse pass first, then AutoReject on survivors
ARTIFACT_REJECTION_METHOD = 'peak_to_peak'
PEAK_TO_PEAK_THRESH = 120e-6   # volts; only used when method is 'peak_to_peak' or 'both'

# ── Raw clean I/O ─────────────────────────────────────────────────────────────
# Saves cleaned continuous data after re-referencing (before epoching).
# Useful checkpoint: re-run epoching/rejection without redoing filtering/ICA.
SAVE_RAW_CLEAN           = False
LOAD_RAW_CLEAN_IF_EXISTS = True   # set False to force re-run from raw

# ── Epochs I/O ───────────────────────────────────────────────────────────────
# SAVE_EPOCHS      — write clean epochs to disk after artifact rejection
# LOAD_EPOCHS_IF_EXISTS — if the file already exists, skip raw→epochs pipeline
#                         and load directly (avoids re-running AutoReject, etc.)
SAVE_EPOCHS           = False
LOAD_EPOCHS_IF_EXISTS = True   # set False to force a full re-run

# ── Occipital channels for SSVEP ──────────────────────────────────────────────
OCCIPITAL_REGEXP = r'E8[1-4]|E8[8-9]|E90|E91|E94|E95' # right hemi occipital channels

# ── SSVEP SNR ─────────────────────────────────────────────────────────────────
SSVEP_STIM_FREQS  = (5.0, 6.0, 7.5)   # fundamental stimulation frequencies (Hz)
SSVEP_FMAX        = 75.0               # highest harmonic to include (Hz)
SSVEP_N_NEIGHBORS = 3                  # noise bins collected on each side per harmonic
SSVEP_SNR_TYPE    = 'amplitude_ratio'            # 'ratio' (power), 'amplitude_ratio', or 'zscore'
SSVEP_N_EPOCHS    = None               # int → subsample to this many core epochs; None → use all


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_fig(report, fig, title, section):
    """Add a figure to the report and optionally display it."""
    if report is not None:
        report.add_figure(fig, title=title, section=section)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_fft_histogram(epochs, picks=None, fmax=80,
                       target_freqs=[10, 12, 15, 20, 24, 30, 36, 40, 45, 48, 60],
                       title='FFT Amplitude Histogram'):
    """
    Plot FFT amplitude spectrum at native frequency resolution.

    average complex FFT values across epochs + channels
    """
    picks = picks or 'eeg'
    data = epochs.get_data(picks=picks)   # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_times = data.shape[-1]

    freqs = np.fft.rfftfreq(n_times, d=1. / sfreq)   # native frequency bins

    # Average complex FFT first → then amplitude (noise cancels, signal survives)
    complex_fft = np.fft.rfft(data, axis=-1)          # (n_epochs, n_channels, n_freqs)
    mean_amp = np.abs(complex_fft.mean(axis=(0, 1))) * 1e6   # V → µV

    # Trim to fmax
    mask = freqs <= fmax
    plot_freqs = freqs[mask]
    plot_amp   = mean_amp[mask]

    freq_res = freqs[1] - freqs[0]   # bin width (e.g. 0.5 Hz for 2 s epochs)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(plot_freqs, plot_amp, width=freq_res * 0.9,
           color='steelblue', alpha=0.7, align='center')

    for f in (target_freqs or []):
        ax.axvline(f, color='red', linestyle='--', alpha=0.7, linewidth=1.2, label=f'{f} Hz')

    # Label every 5 Hz on the x-axis; target lines mark the exact frequencies
    tick_step = 5
    ax.set_xticks(np.arange(0, fmax + tick_step, tick_step))
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (µV)',
           title=title, xlim=[-freq_res / 2, fmax + freq_res])
    plt.tight_layout()
    return fig, ax


def compute_ssvep_snr(epochs_or_evoked, stim_freqs=(5., 6., 7.5), fmax=75.,
                      n_neighbors=1, snr_type='ratio', picks='eeg'):
    """Compute per-channel SSVEP SNR from even harmonics vs. neighboring bins.

    Averages epochs in the time domain before FFT.
    Power at each harmonic compared to the local noise floor estimated from adjacent bins (skipping any bin that
    coincides with a harmonic of any stimulus frequency, and skipping 60 Hz).

    Parameters
    ----------
    epochs_or_evoked : mne.Epochs | mne.Evoked
    stim_freqs : tuple[float]
        Fundamental stimulus frequencies (Hz).  Even harmonics 2f, 4f, … are used.
    fmax : float
        Highest harmonic frequency to include (Hz).
    n_neighbors : int
        Number of noise bins on each side of each harmonic bin (default 1).
    snr_type : 'ratio' | 'amplitude_ratio' | 'zscore'
        'ratio' (default): Σ P(f_h) / Σ μ_P(f_h).  Power ratio; signal is N× the
            noise floor in power.  Dimensionless → directly comparable across EEG
            and MEG sessions with the same task.
        'amplitude_ratio': Σ |X(f_h)| / Σ μ_A(f_h).  Same logic on amplitude
            (square root of power).  Norcia-lab convention; slightly more intuitive
            ("signal amplitude is N× noise amplitude").  Also cross-modality
            comparable.  Numerically ≈ √(power_ratio) but not identical because
            of how the summation interacts with the square root.
        'zscore': Σ_h (P(f_h) − μ_noise_h) / μ_noise_h  =  Σ(ratio_h − 1).
            Normalized excess; harder to interpret because the sum grows with the
            number of harmonics (5 Hz gets 6 terms, 7.5 Hz gets 3).
    picks : str | list
        Channel selection.

    Returns
    -------
    snr : dict[float → ndarray(n_channels,)]
        SNR per channel for each stimulus frequency.
    info : mne.Info
        Channel info matching the returned arrays.
    harmonics_used : dict[float → list[float]]
        Actual harmonic frequencies analyzed per stim freq (for diagnostics).
    """
    # ── Time-domain average → single evoked signal ────────────────────────────
    # Averaging first (before FFT)
    if isinstance(epochs_or_evoked, mne.BaseEpochs):
        evoked = epochs_or_evoked.average(picks=picks)
    else:
        evoked = epochs_or_evoked.copy().pick(picks)

    data = evoked.data           # (n_channels, n_times)  real-valued
    sfreq = evoked.info['sfreq']
    n_times = data.shape[-1]
    n_ch = data.shape[0]

    # FFT on the time-averaged signal; power = |X[k]|²
    # np.fft.rfft returns the one-sided spectrum.
    # All bins share the same normalization so ratios and z-scores are exact.
    X = np.fft.rfft(data, axis=-1)      # (n_ch, n_freqs)  complex128
    A = np.abs(X)                        # (n_ch, n_freqs)  amplitude
    P = A ** 2                           # (n_ch, n_freqs)  power; same scale for all bins
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    freq_res = freqs[1] - freqs[0]

    # Enumerate valid even harmonics
    all_harmonic_bins = set()   # bins to never use as noise neighbors
    stim_harmonics = {}         # sf → [(bin_idx, freq), ...]  harmonics to analyze

    for sf in stim_freqs:
        hlist = []
        mult = 2
        while True:
            f_h = sf * mult
            if f_h > fmax + freq_res * 0.5:
                break
            k_h = int(round(f_h / freq_res))
            if k_h >= len(freqs):
                mult += 2
                continue
            actual_f = freqs[k_h]
            all_harmonic_bins.add(k_h)
            # Skip 60 Hz (notch-filtered; still exclude from noise bins)
            if abs(actual_f - 60.0) < freq_res * 0.5:
                mult += 2
                continue
            if actual_f > fmax:
                mult += 2
                continue
            # Warn if epoch length is mismatched and harmonic lands off-bin
            if abs(f_h - actual_f) > freq_res * 0.01:
                print(f"  Warning: {sf} Hz × {mult} = {f_h:.3f} Hz → nearest bin "
                      f"{actual_f:.3f} Hz (offset {abs(f_h-actual_f):.4f} Hz; "
                      f"freq_res={freq_res:.4f} Hz).  Check epoch length.")
            hlist.append((k_h, actual_f))
            mult += 2
        stim_harmonics[sf] = hlist

    harmonics_used = {sf: [f for _, f in v] for sf, v in stim_harmonics.items()}

    # ── Helper: find n valid neighbor bins on each side, skipping harmonics ───
    def _neighbor_bins(k_h, n, excluded, max_k):
        left, right = [], []
        k = k_h - 1
        while len(left) < n and k >= 0:
            if k not in excluded:
                left.append(k)
            k -= 1
        k = k_h + 1
        while len(right) < n and k < max_k:
            if k not in excluded:
                right.append(k)
            k += 1
        return left + right

    # ── SNR per stimulus frequency ────────────────────────────────────────────
    snr = {}
    for sf, hlist in stim_harmonics.items():
        if not hlist:
            snr[sf] = np.zeros(n_ch)
            continue

        if snr_type in ('ratio', 'amplitude_ratio'):
            # Σ signal / Σ mean_noise — power for 'ratio', amplitude for 'amplitude_ratio'
            vals = P if snr_type == 'ratio' else A
            sig_sum = np.zeros(n_ch)
            noi_sum = np.zeros(n_ch)
            for k_h, _ in hlist:
                nbr = _neighbor_bins(k_h, n_neighbors, all_harmonic_bins, len(freqs))
                if not nbr:
                    continue
                sig_sum += vals[:, k_h]
                noi_sum += vals[:, nbr].mean(axis=1)
            snr[sf] = np.where(noi_sum > 0, sig_sum / noi_sum, 0.0)

        else:  # 'zscore' — Σ (ratio_h − 1) across harmonics
            # Denominator is the local noise mean, not σ.  Using σ collapses when
            # n_neighbors is small: a smooth noise floor gives σ ≈ 0 for just 2 bins,
            # inflating z into the thousands.  Dividing by μ gives (P/μ − 1), i.e.
            # "how many noise-floor units above noise" — stable, bounded below by -1.
            z_accum = np.zeros(n_ch)
            for k_h, _ in hlist:
                nbr = _neighbor_bins(k_h, n_neighbors, all_harmonic_bins, len(freqs))
                if not nbr:
                    continue
                mu = P[:, nbr].mean(axis=1)
                mu = np.where(mu > 0, mu, np.finfo(float).eps)
                z_accum += (P[:, k_h] - mu) / mu          # = ratio_h − 1
            snr[sf] = z_accum

    return snr, evoked.info, harmonics_used


def plot_ssvep_snr_topo(snr_dict, info, harmonics_used=None, snr_type='zscore'):
    """Sensor topography of SSVEP SNR, one subplot per stimulus frequency."""
    stim_freqs = sorted(snr_dict.keys())
    n = len(stim_freqs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    cbar_label = {'ratio': 'SNR (power ratio)',
                  'amplitude_ratio': 'SNR (amplitude ratio)',
                  'zscore': 'SNR (Σ ratio−1)'}.get(snr_type, 'SNR')

    for ax, sf in zip(axes, stim_freqs):
        data = snr_dict[sf]
        vmin, vmax = np.percentile(data, [5, 95])
        # Symmetric color range around zero for z-score; asymmetric for ratio
        if snr_type == 'zscore':
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False,
                                     cmap='RdBu_r', vlim=(vmin, vmax))
        h_str = ', '.join(f'{h:.1f}' for h in (harmonics_used or {}).get(sf, []))
        ax.set_title(f'{sf} Hz\n[{h_str} Hz]', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.suptitle(f'SSVEP SNR Topography ({snr_type})', y=1.02)
    plt.tight_layout()
    return fig


def report_ssvep_snr(epochs_clean, stim_freqs=None, fmax=None, n_neighbors=None,
                     snr_type=None, n_epochs=None, picks='eeg', report=None):
    """Compute SSVEP SNR on core epochs and add topography to the report.

    Only 'bin/*' epochs are used (prelude/postlude noise epochs are excluded).
    Optionally subsamples to n_epochs core epochs (random, without replacement)
    to assess how SNR scales with trial count.
    """
    sec = '8 · SSVEP SNR'
    stim_freqs  = stim_freqs  if stim_freqs  is not None else SSVEP_STIM_FREQS
    fmax        = fmax        if fmax        is not None else SSVEP_FMAX
    n_neighbors = n_neighbors if n_neighbors is not None else SSVEP_N_NEIGHBORS
    snr_type    = snr_type    if snr_type    is not None else SSVEP_SNR_TYPE
    n_epochs    = n_epochs    if n_epochs    is not None else SSVEP_N_EPOCHS

    # ── Restrict to core stimulus epochs only ─────────────────────────────────
    core = epochs_clean['bin']
    n_available = len(core)

    # ── Optional random subsample ─────────────────────────────────────────────
    if n_epochs is not None:
        if n_epochs > n_available:
            print(f"  Warning: requested {n_epochs} epochs but only {n_available} "
                  f"core epochs available; using all.")
            n_epochs = n_available
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n_available, size=n_epochs, replace=False)
        idx.sort()
        core = core[idx]
        epoch_label = f'{n_epochs}/{n_available} core epochs (random subsample)'
    else:
        epoch_label = f'{n_available} core epochs'

    print(f"  SSVEP SNR: using {epoch_label}")

    snr_dict, info, harmonics_used = compute_ssvep_snr(
        core, stim_freqs=stim_freqs, fmax=fmax,
        n_neighbors=n_neighbors, snr_type=snr_type, picks=picks,
    )

    for sf in sorted(stim_freqs):
        arr = snr_dict[sf]
        print(f"  SSVEP SNR [{snr_type}] {sf} Hz: "
              f"mean={arr.mean():.2f}  max={arr.max():.2f}  "
              f"harmonics={harmonics_used[sf]}")

    title = f'SSVEP SNR Topography ({snr_type}) — {epoch_label}'
    fig = plot_ssvep_snr_topo(snr_dict, info, harmonics_used, snr_type=snr_type)
    _add_fig(report, fig, title, sec)

    return snr_dict, info


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · Raw Data
# ══════════════════════════════════════════════════════════════════════════════

def report_raw(raw, report=None):
    """Add raw data visualisations to the report."""
    sec = '1 · Raw Data'

    if report is not None:
        report.add_raw(raw, title='Raw Data', psd=True, butterfly=False)

    # Sensor layout
    fig = raw.plot_sensors(show_names=True, show=False)
    _add_fig(report, fig, 'Sensor Layout', sec)

    # Per-channel variance — outliers → bad channels
    data, _ = raw.get_data(return_times=True)
    chan_var = np.var(data, axis=1)
    eeg_idx  = mne.pick_types(raw.info, eeg=True)
    fig, ax  = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(eeg_idx)), chan_var[eeg_idx], width=1.0)
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Variance (V²)')
    ax.set_title('Per-channel variance (EEG only) — outliers → bad channels')
    plt.tight_layout()
    _add_fig(report, fig, 'Channel Variance', sec)

    # Events summary
    events = mne.find_events(raw, verbose=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                        first_samp=raw.first_samp, axes=ax, show=False)
    plt.tight_layout()
    _add_fig(report, fig, 'All Events', sec)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · Filtering
# ══════════════════════════════════════════════════════════════════════════════

def run_filtering(raw, l_freq=0.5, h_freq=70, report=None):
    """Bandpass + 60 Hz notch filter.  Returns filtered raw.

    Parameters
    ----------
    l_freq : float
        High-pass cutoff (Hz).  Removes slow drifts below this frequency.
    h_freq : float
        Low-pass cutoff (Hz).  Removes high-frequency noise above this frequency.
    """
    sec = '2 · Filtering'
    filter_label = f'{l_freq}–{h_freq} Hz + 60 Hz notch'

    if report is not None:
        # PSD before filtering
        n_fft = int(raw.info['sfreq'] * 10)
        fig_pre = raw.compute_psd(fmin=0, fmax=h_freq, picks='eeg', n_fft=n_fft).plot(
            average=False, spatial_colors=True, show=False)
        _add_fig(report, fig_pre, 'PSD — Before Filtering', sec)

    raw_fil = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    raw_fil.notch_filter(freqs=60)

    if report is not None:
        n_fft = int(raw.info['sfreq'] * 10)
        # PSD after filtering
        fig_post = raw_fil.compute_psd(fmin=0, fmax=h_freq, picks='eeg', n_fft=n_fft).plot(
            average=False, spatial_colors=True, show=False)
        _add_fig(report, fig_post, f'PSD — After Filtering ({filter_label})', sec)


    return raw_fil


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · Bad Channels
# ══════════════════════════════════════════════════════════════════════════════

def run_bad_channels(raw, bad_channels, interpolate=True, report=None):
    """Mark bad channels, optionally interpolate them, and visualise.

    Interpolation (spherical spline) reconstructs the bad channel from its
    neighbors so it contributes a valid signal to the average reference.
    Should be done before re-referencing.  Returns raw (in-place).
    """
    sec = '3 · Bad Channels'
    raw.info['bads'] = bad_channels

    if report is not None:
        fig = raw.plot_sensors(show_names=True, show=False)
        _add_fig(report, fig, 'Sensor Layout — Bad Channels Marked (red)', sec)

    if interpolate and bad_channels:
        raw.interpolate_bads(reset_bads=True)   # reset_bads clears info['bads'] after interpolation
        print(f"Interpolated {len(bad_channels)} bad channel(s): {bad_channels}")

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · ICA
# ══════════════════════════════════════════════════════════════════════════════

def run_ica(raw, exclude_comps, eog_proxy_ch, n_components=25, report=None):
    """Fit ICA, inspect components, apply to raw.  Returns cleaned raw copy."""
    sec = '4 · ICA'

    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    ica.fit(raw)

    if report is not None:
        # Topographic maps of all components
        figs_comp = ica.plot_components(show=False)
        figs_comp = figs_comp if isinstance(figs_comp, list) else [figs_comp]
        for i, fig in enumerate(figs_comp):
            _add_fig(report, fig, f'ICA Components (page {i + 1})', sec)

        # Properties of each excluded component
        figs_prop = ica.plot_properties(raw, picks=exclude_comps, show=False)
        figs_prop = figs_prop if isinstance(figs_prop, list) else [figs_prop]
        for i, fig in enumerate(figs_prop):
            _add_fig(report, fig,
                     f'ICA Component {exclude_comps[i % len(exclude_comps)]} Properties', sec)

        # EOG scores
        _, eog_scores = ica.find_bads_eog(raw, ch_name=eog_proxy_ch)
        fig_scores = ica.plot_scores(eog_scores, show=False)
        _add_fig(report, fig_scores,
                 f'ICA EOG Scores (proxy: {eog_proxy_ch})', sec)

    ica.exclude = exclude_comps

    if report is not None:
        # Overlay before vs after
        fig_ov = ica.plot_overlay(raw, exclude=ica.exclude, show=False)
        _add_fig(report, fig_ov,
                 'ICA Overlay — Before vs After Artifact Removal', sec)

    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · Re-referencing
# ══════════════════════════════════════════════════════════════════════════════

def run_rereferencing(raw, report=None):
    """Apply average reference.  Returns raw (in-place)."""
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()

    if report is not None:
        sec = '4 · Re-referencing'
        n_fft = int(raw.info['sfreq'] * 10)
        fig = raw.compute_psd(fmin=0, fmax=50, picks='eeg', n_fft=n_fft).plot(
            average=False, spatial_colors=True, show=False)
        _add_fig(report, fig, 'PSD — After Re-referencing', sec)

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · Epoching
# ══════════════════════════════════════════════════════════════════════════════

def run_epoching(raw, raw_clean, tmin, tmax, core_only=False, report=None):
    """Build epochs from DIN4/DIN5 events.  Returns Epochs object.

    Trial structure (each segment is bin_duration seconds):
        DIN4 ── prelude ── DIN5[0..4] (5 core bins) ── DIN5[5] postlude ── DIN5[6] ISI …

    Event IDs
    ---------
    bin/0 … bin/4  : 0–4   core stimulus bins (always included)
    noise/prelude  : 5     pre-stimulus period (DIN4 → DIN5[0])
    noise/postlude : 6     post-stimulus period (DIN5[5] → DIN5[6])

    Parameters
    ----------
    core_only : bool
        False (default) — include prelude + postlude epochs labeled as
        'noise/prelude' / 'noise/postlude' so ``epochs['noise']`` selects
        both for noise-covariance estimation.
        True — only 'bin/0'–'bin/4' (original behaviour).
    """
    sec = '5 · Epoching'
    sfreq = raw_clean.info['sfreq']

    # Compute tmax so the epoch contains exactly the intended number of samples.
    # Floating-point arithmetic (e.g. 2.0 - 1/1000) can land a hair above or
    # below a sample boundary, causing MNE to produce n±1 samples.  Rounding
    # in sample space first and converting back avoids this.
    n_samples  = int(round((tmax - tmin) * sfreq))   # e.g. 2000
    tmax_exact = tmin + (n_samples - 1) / sfreq      # last sample, no rounding error

    din4_events = mne.find_events(raw, stim_channel='DIN4')   # trial starts
    din5_events = mne.find_events(raw, stim_channel='DIN5')   # bin onsets

    din4_samples = din4_events[:, 0]
    din5_samples = din5_events[:, 0]

    event_id = {
        'bin/0': 0, 'bin/1': 1, 'bin/2': 2, 'bin/3': 3, 'bin/4': 4,
        'noise/prelude':  5,
        'noise/postlude': 6,
    }

    all_events = []
    for i, trial_start in enumerate(din4_samples):
        trial_end       = din4_samples[i + 1] if i + 1 < len(din4_samples) else np.inf
        dins_in_trial   = din5_samples[(din5_samples >= trial_start) &
                                       (din5_samples <  trial_end)]

        # Core bins: DIN5 positions 0–4 → event IDs 0–4
        for pos, s in enumerate(dins_in_trial[:5]):
            all_events.append([s, 0, pos])

        if not core_only:
            # Prelude: epoch anchored at DIN4, lasts bin_duration seconds
            if len(dins_in_trial) >= 1:
                all_events.append([trial_start, 0, 5])

            # Postlude: DIN5[5] (6th pulse, 0-based index 5) → DIN5[6] onset
            # DIN5[6] is the ISI — we do NOT epoch it, just use [5] as anchor
            if len(dins_in_trial) >= 6:
                all_events.append([dins_in_trial[5], 0, 6])

    all_events = np.array(all_events)
    all_events = all_events[np.argsort(all_events[:, 0])]   # sort by sample

    epochs = mne.Epochs(raw_clean, all_events, event_id=event_id,
                        tmin=tmin, tmax=tmax_exact,
                        baseline=None, preload=True)

    if report is not None:
        # Report figures are computed on core bins only for interpretability
        core_epochs = epochs['bin']

        # Butterfly + topomaps at peak time points
        figs_avg = core_epochs.average().plot_joint(show=False)
        figs_avg = figs_avg if isinstance(figs_avg, list) else [figs_avg]
        for fig in figs_avg:
            _add_fig(report, fig, 'Epoch Average — Before Rejection (core bins)', sec)

        fig_epsd = core_epochs.compute_psd(fmin=0, fmax=50, method='welch',
                                           n_fft=len(core_epochs.times)).plot(show=False)
        _add_fig(report, fig_epsd, 'Epochs PSD (0–50 Hz)', sec)

    return epochs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · Artifact Rejection
# ══════════════════════════════════════════════════════════════════════════════

def _reject_peak_to_peak(epochs, thresh, report=None, sec='6 · Artifact Rejection'):
    """Drop epochs exceeding a fixed peak-to-peak amplitude threshold."""
    epochs_out = epochs.copy()
    epochs_out.drop_bad(reject=dict(eeg=thresh))

    print(f"Peak-to-peak: dropped {len(epochs) - len(epochs_out)} / {len(epochs)} epochs "
          f"(threshold ±{int(thresh * 1e6)} µV)")

    if report is not None:
        fig = epochs_out.plot_drop_log(show=False)
        _add_fig(report, fig,
                 f'Drop Log — Peak-to-Peak (±{int(thresh * 1e6)} µV)', sec)

    return epochs_out


def _reject_autoreject(epochs, report=None, sec='6 · Artifact Rejection'):
    """Data-driven rejection and interpolation via AutoReject."""
    ar = AutoReject()
    epochs_out, reject_log = ar.fit_transform(epochs, return_log=True)

    labels           = reject_log.labels
    n_interpolated   = (labels == 1).sum()
    n_epochs_dropped = reject_log.bad_epochs.sum()

    print(f"AutoReject: dropped {n_epochs_dropped} epochs, "
          f"interpolated {n_interpolated} channel-epoch pairs")

    ch_interp_counts = (labels == 1).sum(axis=0)
    top_channels = np.argsort(ch_interp_counts)[::-1][:10]
    for idx in top_channels:
        if ch_interp_counts[idx] > 0:
            print(f"  {epochs.ch_names[idx]}: interpolated in {ch_interp_counts[idx]} epochs")

    if report is not None:
        fig_log = reject_log.plot('horizontal', show=False)
        _add_fig(report, fig_log, 'AutoReject Log (horizontal)', sec)

    return epochs_out


def run_artifact_rejection(epochs, method, peak_to_peak_thresh, report=None):
    """Dispatch to the chosen rejection method.  Returns clean Epochs.

    method:
        'autoreject'   — data-driven (recommended default)
        'peak_to_peak' — fixed amplitude threshold
        'both'         — peak-to-peak coarse pass, then AutoReject on survivors
    """
    sec = '6 · Artifact Rejection'

    if method == 'peak_to_peak':
        epochs_clean = _reject_peak_to_peak(epochs, peak_to_peak_thresh, report, sec)
        label = 'After Peak-to-Peak Rejection'

    elif method == 'autoreject':
        epochs_clean = _reject_autoreject(epochs, report, sec)
        label = 'After AutoReject'

    elif method == 'both':
        epochs_pp    = _reject_peak_to_peak(epochs, peak_to_peak_thresh, report, sec)
        epochs_clean = _reject_autoreject(epochs_pp, report, sec)
        label = 'After Peak-to-Peak + AutoReject'

    else:
        raise ValueError(f"Unknown ARTIFACT_REJECTION_METHOD: {method!r}. "
                         "Choose 'autoreject', 'peak_to_peak', or 'both'.")

    if report is not None:
        # Per-condition epoch counts
        n_total   = len(epochs)
        n_kept    = len(epochs_clean)
        fig_cnt, ax_cnt = plt.subplots(figsize=(8, 4))
        conditions = ['bin/0', 'bin/1', 'bin/2', 'bin/3', 'bin/4',
                      'noise/prelude', 'noise/postlude']
        counts_before = [epochs[c].events.shape[0] if c in epochs.event_id else 0
                         for c in conditions]
        counts_after  = [epochs_clean[c].events.shape[0] if c in epochs_clean.event_id else 0
                         for c in conditions]
        x = np.arange(len(conditions))
        ax_cnt.bar(x - 0.2, counts_before, 0.4, label='Before rejection', color='steelblue', alpha=0.7)
        ax_cnt.bar(x + 0.2, counts_after,  0.4, label='After rejection',  color='darkorange', alpha=0.7)
        for xi, (cb, ca) in enumerate(zip(counts_before, counts_after)):
            pct = 100 * ca / cb if cb > 0 else 0
            ax_cnt.text(xi, max(cb, ca) + 0.5, f'{ca}/{cb}\n({pct:.0f}%)',
                        ha='center', va='bottom', fontsize=8)
        ax_cnt.set_xticks(x)
        ax_cnt.set_xticklabels([c.replace('noise/', '') for c in conditions])
        ax_cnt.set_ylabel('Epoch count')
        ax_cnt.set_title(f'Epochs retained per condition — {label}\n'
                         f'Total: {n_kept}/{n_total} kept ({100*n_kept/n_total:.0f}%)')
        ax_cnt.legend()
        plt.tight_layout()
        _add_fig(report, fig_cnt, f'Epoch Counts per Condition — {label}', sec)

        # Butterfly + topomaps at peak time points (core bins only)
        figs_cavg = epochs_clean['bin'].average().plot_joint(show=False)
        figs_cavg = figs_cavg if isinstance(figs_cavg, list) else [figs_cavg]
        for fig in figs_cavg:
            _add_fig(report, fig, f'Epoch Average — {label}', sec)

        fig_psd = epochs_clean.compute_psd(fmin=0, fmax=50, method='welch',
                                           n_fft=len(epochs_clean.times)).plot(show=False)
        _add_fig(report, fig_psd, f'Epochs PSD (0–50 Hz) — {label}', sec)

    return epochs_clean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · Occipital SSVEP PSD
# ══════════════════════════════════════════════════════════════════════════════

def report_ssvep_psd(epochs_clean, occipital_regexp, report=None):
    """Plot occipital PSD and FFT histograms and add to report."""
    sec = '7 · Occipital SSVEP PSD'
    occipital_picks = mne.pick_channels_regexp(epochs_clean.ch_names, occipital_regexp)

    # Welch PSD (all occipital channels)
    psd = epochs_clean.compute_psd(fmin=0, fmax=50, picks=occipital_picks,
                                   method='welch', n_fft=len(epochs_clean.times))
    fig_opsd = psd.plot(show=False)
    _add_fig(report, fig_opsd,
             'Occipital PSD — Clean Epochs (0–50 Hz)\n'
             'Expected SSVEP harmonics: 10, 12, 15, 20, 24, 30 Hz',
             sec)

    # FFT histogram — occipital channels averaged
    fig_fft_occ, _ = plot_fft_histogram(
        epochs_clean, picks=occipital_picks,
        title='FFT — Occipital channels average')
    _add_fig(report, fig_fft_occ, 'FFT Histogram — Occipital Channels', sec)

    # FFT histogram — single representative sensor E90
    if 'E90' in epochs_clean.ch_names:
        fig_fft_e90, _ = plot_fft_histogram(
            epochs_clean, picks=['E90'],
            title='FFT — single occipital sensor')
        _add_fig(report, fig_fft_e90, 'FFT Histogram — E90', sec)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    raw_clean_path = os.path.expanduser(
        f"{data_path}/{subject_id}/eeg/{subject_id}_preproc-raw.fif"
    )
    epochs_fif_path = os.path.expanduser(
        f"{data_path}/{subject_id}/eeg/{subject_id}_preproc-epoch.fif"
    )

    # ── Create report ─────────────────────────────────────────────────────────
    report = mne.Report(title=f"EEG Preprocessing — {subject_id}", verbose=True) \
             if SAVE_REPORT else None

    # ── Fast path A: load cached clean epochs (skip everything) ───────────────
    if LOAD_EPOCHS_IF_EXISTS and os.path.exists(epochs_fif_path):
        print(f"Loading cached clean epochs from {epochs_fif_path}")
        epochs_clean = mne.read_epochs(epochs_fif_path, preload=True)
        raw = None   # not needed downstream

    else:
        # ── Fast path B: load cached clean raw (skip filtering/ICA/reref) ──────
        if LOAD_RAW_CLEAN_IF_EXISTS and os.path.exists(raw_clean_path):
            print(f"Loading cached clean raw from {raw_clean_path}")
            raw_clean = mne.io.read_raw_fif(raw_clean_path, preload=True)
            # Also need original raw for event detection in run_epoching
            raw = mne.io.read_raw_egi(
                f"{data_path}/{subject_id}/eeg/KHNCL-NetStation/"
                f"{subject_id}_V1Loc_2026{recording_str}.mff",
                preload=True   # need data for report figures
            )
            raw.set_channel_types({'VREF': 'misc'})

            # Still generate sections 1–4b for the report from the loaded data
            report_raw(raw, report)
            if DO_FILTERING:
                run_filtering(raw, l_freq=FILTER_HIGH, h_freq=FILTER_LOW, report=report)
            if DO_BAD_CHANNELS:
                run_bad_channels(raw.copy(), BAD_CHANNELS, INTERPOLATE_BADS, report)
            if DO_ICA:
                run_ica(raw_clean, ICA_EXCLUDE_COMPS, ICA_EOG_PROXY_CH,
                        n_components=ICA_N_COMPONENTS, report=report)
            if DO_REREFERENCING:
                run_rereferencing(raw_clean.copy(), report)

        else:
            # ── Full pipeline from raw ─────────────────────────────────────────
            raw = mne.io.read_raw_egi(
                f"{data_path}/{subject_id}/eeg/KHNCL-NetStation/"
                f"{subject_id}_V1Loc_2026{recording_str}.mff",
                preload=True
            )
            raw.set_channel_types({'VREF': 'misc'})

            # ── Section 1 · Raw Data ───────────────────────────────────────────
            report_raw(raw, report)

            # ── Section 2 · Filtering ──────────────────────────────────────────
            raw_fil = run_filtering(raw, l_freq=FILTER_HIGH, h_freq=FILTER_LOW, report=report) if DO_FILTERING else raw.copy()

            # ── Section 3 · Bad Channels ───────────────────────────────────────
            if DO_BAD_CHANNELS:
                run_bad_channels(raw_fil, BAD_CHANNELS, INTERPOLATE_BADS, report)

            # ── Section 4 · ICA ────────────────────────────────────────────────
            if DO_ICA:
                raw_clean = run_ica(raw_fil, ICA_EXCLUDE_COMPS, ICA_EOG_PROXY_CH,
                                    n_components=ICA_N_COMPONENTS, report=report)
            else:
                raw_clean = raw_fil.copy()

            # ── Section 4b · Re-referencing ────────────────────────────────────
            if DO_REREFERENCING:
                run_rereferencing(raw_clean, report)

            # ── Save clean raw ─────────────────────────────────────────────────
            if SAVE_RAW_CLEAN:
                raw_clean.save(raw_clean_path, overwrite=True)
                print(f"Clean raw saved → {raw_clean_path}")

        # ── Section 5 · Epoching ───────────────────────────────────────────────
        if DO_EPOCHING:
            epochs = run_epoching(raw, raw_clean, EPOCH_TMIN, EPOCH_TMAX,
                                  core_only=EPOCH_CORE_ONLY, report=report)
        else:
            epochs = None

        # ── Section 6 · Artifact Rejection ─────────────────────────────────────
        if DO_ARTIFACT_REJECTION and epochs is not None:
            epochs_clean = run_artifact_rejection(epochs, ARTIFACT_REJECTION_METHOD,
                                                  PEAK_TO_PEAK_THRESH, report)
        else:
            epochs_clean = epochs

        # ── Save clean epochs ──────────────────────────────────────────────────
        if SAVE_EPOCHS and epochs_clean is not None:
            epochs_clean.save(epochs_fif_path, overwrite=True)
            print(f"Clean epochs saved → {epochs_fif_path}")

    # ── Section 7 · Occipital SSVEP PSD ──────────────────────────────────────
    if DO_SSVEP_PSD and epochs_clean is not None:
        report_ssvep_psd(epochs_clean, OCCIPITAL_REGEXP, report)

    # ── Section 8 · SSVEP SNR ─────────────────────────────────────────────────
    if DO_SSVEP_SNR and epochs_clean is not None:
        report_ssvep_snr(epochs_clean, report=report)

    # ── Save report ───────────────────────────────────────────────────────────
    if SAVE_REPORT and report is not None:
        report_path = os.path.expanduser(
            f"{data_path}/{subject_id}/eeg/{subject_id}_preproc_report.html"
        )
        report.save(report_path, overwrite=True, open_browser=False)
        print(f"\nReport saved → {report_path}")


if __name__ == '__main__':
    main()
