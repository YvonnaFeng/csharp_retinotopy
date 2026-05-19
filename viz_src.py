"""Visualization helpers for SSVEP source-localization results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from mne.surface import read_surface


_DEFAULT_OCC_CH = (
    "E81", "E82", "E83", "E84", "E88", "E89", "E90", "E91", "E94", "E95",  # EGI HydroCel
)

# hemi and size are intentionally absent — both are auto-derived from the hemi parameter
_BRAIN_DEFAULTS = dict(
    smoothing_steps=3,
    background="white",
    foreground="black",
)


def _auto_brain_size(hemi):
    """Single-column layout (one hemisphere, two views stacked) vs two-column split."""
    return (500, 800) if hemi in ("rh", "lh") else (800, 800)

def _auto_foci_scale(hemi):
    return 1.2 if hemi in ("rh", "lh") else 0.6

def _auto_cbar_height(hemi):
    return 0.08 if hemi in ("rh", "lh") else 0.04


def viz_filtered_stcs(
    filtered_stcs,
    freq_map,
    harmonics_map,
    subjects_dir,
    inverse_method="dSPM",
    visual_labels=None,
    mode="snapshot",
    n_snapshots=5,
    hemi="split",
    views=("caudal", "medial"),
    show_evoked=None,
    occ_channels=None,
    save_dir=None,
    marker="peak",
    com_top_num=20,
    foci_scale=None,
    **brain_kwargs,
):
    """Visualize filtered STC results as per-cycle snapshots or a video.

    Parameters
    ----------
    filtered_stcs : dict
        Output of get_filtered_stc: {label: {"stc": SourceEstimate, "evoked": Evoked}}.
    freq_map : dict
        {label: base_freq_hz}.
    harmonics_map : dict
        {label: harmonics_type} where harmonics_type is "even", "odd", or "all".
    subjects_dir : str
        FreeSurfer subjects directory.
    inverse_method : str
        Label used for output file naming.
    visual_labels : list of (mne.Label, str) or None
        Anatomical labels to overlay; each item is (label_obj, color_str).
    mode : {"snapshot", "video"}
        "snapshot" saves N evenly-spaced PNG files within one SSVEP cycle.
        "video" saves a 0.5-second movie at 0.2× real-time (5× time dilation).
    n_snapshots : int
        Number of snapshots within one cycle (mode="snapshot" only).
    hemi : {"split", "rh", "lh"}
        Which hemisphere(s) to render. "rh" and "lh" produce a compact single-column
        layout; "split" produces the two-column side-by-side layout.
        Size, colorbar thickness, and foci scale are auto-adjusted unless overridden.
    views : sequence of str
        Brain surface views, e.g. ("caudal", "medial").
    show_evoked : bool or None
        Plot filtered evoked diagnostic with cycle-selection overlay.
        None → True for snapshot, False for video.
    occ_channels : list of str or None
        Channel names for the evoked diagnostic. None = auto-detect standard
        occipital channels; falls back to all channels if none are found.
    save_dir : str or None
        Directory to write output files. None = show interactively instead of saving.
    marker : {"peak", "com", None}
        How to place the focus marker on the rh brain.
        "peak"  — single vertex with maximum activation at that time point.
        "com"   — weighted centre-of-mass of the top ``com_top_num`` vertices,
                  snapped to the nearest source-space vertex on rh.mid.
        None    — no marker.
    com_top_num : int
        Number of top-activated rh vertices used to compute the COM centroid
        (only relevant when marker="com").
    foci_scale : float or None
        Scale factor for the focus sphere marker. None = auto (1.2 for single-hemi,
        0.6 for split).
    **brain_kwargs
        Forwarded to stc.plot(), overriding built-in defaults (background, size, etc.).
    """
    if visual_labels is None:
        visual_labels = []
    if show_evoked is None:
        show_evoked = (mode == "snapshot")
    if foci_scale is None:
        foci_scale = _auto_foci_scale(hemi)

    # size: use explicit brain_kwargs override, otherwise auto from hemi
    size = brain_kwargs.pop("size", None) or _auto_brain_size(hemi)
    brain_cfg = {**_BRAIN_DEFAULTS, **brain_kwargs, "size": size}

    for label, result in filtered_stcs.items():
        stc            = result["stc"]
        base_freq      = freq_map[label]
        harmonics_type = harmonics_map[label]
        min_order      = 2 if harmonics_type == "even" else 1
        cycle_duration = 1.0 / (min_order * base_freq)

        snap_info = None
        if mode == "snapshot":
            snap_info = _find_cycle_window(
                stc, cycle_duration, n_snapshots, label, base_freq, min_order
            )

        if show_evoked and "evoked" in result:
            _plot_evoked_diagnostic(
                result["evoked"], label, base_freq, harmonics_type,
                inverse_method, save_dir, occ_channels, snap_info,
            )

        if save_dir is not None:
            if mode == "snapshot":
                _render_snapshots(
                    stc, snap_info, label, base_freq, inverse_method,
                    subjects_dir, hemi, views, visual_labels, save_dir, brain_cfg,
                    marker, com_top_num, foci_scale,
                )
            else:
                _render_video(
                    stc, label, base_freq, inverse_method,
                    subjects_dir, hemi, views, visual_labels, save_dir, brain_cfg,
                    marker, com_top_num, foci_scale,
                )
        else:
            _open_interactive(stc, subjects_dir, hemi, views, visual_labels, brain_cfg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_cycle_window(stc, cycle_duration, n_snapshots, label, base_freq, min_order):
    """Return cycle window dict using peak detection on the dominant rh vertex.

    Phase-aligns to a zero-crossing: the window starts T/2 after the first
    detected peak (falling zero → trough → rising zero → approaching next peak).
    This guarantees we stay within [tmin, tmax] without clamping.
    """
    rh_data_all       = stc.rh_data
    dominant_vert_idx = np.argmax(rh_data_all.max(axis=1))
    ts_dominant       = rh_data_all[dominant_vert_idx]
    period_samples    = int(round(cycle_duration / stc.tstep))

    peaks, _ = find_peaks(ts_dominant, distance=int(period_samples * 0.8))

    if len(peaks) > 0:
        t_cycle_start = stc.times[peaks[0]] + cycle_duration / 2
        if t_cycle_start + cycle_duration > stc.times[-1]:
            print(f"  [{label}] Warning: cycle window extends past epoch end — "
                  "snapshots may be clipped.")
    else:
        t_cycle_start = max(stc.tmin, 0.0)
        print(f"  [{label}] Warning: no peaks found, falling back to t={t_cycle_start:.4f}s")

    t_cycle_end    = min(t_cycle_start + cycle_duration, stc.times[-1])
    snapshot_times = np.linspace(t_cycle_start, t_cycle_end, n_snapshots, endpoint=False)

    print(f"  [{label}] Cycle window: {t_cycle_start*1000:.1f}–{t_cycle_end*1000:.1f} ms "
          f"({min_order}×{base_freq} Hz = {1000/cycle_duration:.1f} Hz effective, "
          f"{cycle_duration*1000:.1f} ms/cycle)")

    return {"t_start": t_cycle_start, "t_end": t_cycle_end, "times": snapshot_times}


def _plot_evoked_diagnostic(evoked_filt, label, base_freq, harmonics_type,
                             inverse_method, save_dir, occ_channels, snap_info):
    ch_list = occ_channels or [ch for ch in evoked_filt.ch_names
                                if ch in _DEFAULT_OCC_CH]
    if not ch_list:
        ch_list = evoked_filt.ch_names

    t_ms_all = evoked_filt.times * 1000
    data_occ = evoked_filt.get_data(picks=ch_list) * 1e6  # → µV

    fig, ax = plt.subplots(figsize=(12, 3))
    for ch_data, ch_name in zip(data_occ, ch_list):
        ax.plot(t_ms_all, ch_data, linewidth=0.8, alpha=0.75, label=ch_name)

    if snap_info is not None:
        t0, t1 = snap_info["t_start"], snap_info["t_end"]
        ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.2, color="steelblue",
                   label="selected cycle")
        ax.axvline(t0 * 1000, color="steelblue", linewidth=1.5, linestyle="--")
        for i, snap_t in enumerate(snap_info["times"]):
            ax.axvline(snap_t * 1000, color="crimson", linewidth=0.9, alpha=0.8,
                       label="snapshots" if i == 0 else None)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("µV")
    ax.set_title(f"{label} ({base_freq} Hz, {harmonics_type} harmonics) — filtered evoked")
    fig.tight_layout()

    if save_dir is not None:
        path = os.path.join(save_dir, f"{inverse_method}_{label}_cycle_selection.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{label}] Cycle diagnostic → {path}")
    else:
        plt.show()


def _make_brain(stc, subjects_dir, hemi, views, initial_time, time_viewer, brain_cfg):
    kwargs = {
        **brain_cfg,
        "subjects_dir": subjects_dir,
        "hemi":         hemi,
        "views":        list(views),
        "initial_time": initial_time,
        "time_unit":    "s",
        "time_viewer":  time_viewer,
    }
    brain = stc.plot(**kwargs)
    # Reposition colorbar to bottom; thicker for single-hemi layouts
    sb = brain.plotter.scalar_bar
    sb.SetPosition(0.1, 0.02)
    sb.SetPosition2(0.8, _auto_cbar_height(hemi))
    sb.GetLabelTextProperty().SetFontSize(7)
    sb.GetTitleTextProperty().SetFontSize(7)
    return brain


def _get_com_vertex(stc, t_idx, subjects_dir, com_top_num):
    """Weighted COM of the top-N rh source vertices at t_idx, snapped to the nearest
    source-space vertex on the rh.mid surface.

    Returns (hemi_str, fs_vertex_number) or None if no positive activation.
    Mirrors the logic of compute_peak_com() in analyze_src_loc.py but operates
    on a single time point and stays within rh only.
    """
    rh_data_t = stc.rh_data[:, t_idx]
    if rh_data_t.max() <= 0:
        return None

    rh_src_verts = stc.vertices[1]
    n = min(com_top_num, len(rh_data_t))
    top_idx = np.argsort(rh_data_t)[-n:]          # indices within rh source array
    weights = rh_data_t[top_idx]
    weights = weights / weights.sum()

    surf_path = os.path.join(subjects_dir, stc.subject, "surf", "rh.mid")
    rh_coords, _ = read_surface(surf_path)
    rh_src_coords = rh_coords[rh_src_verts]       # (n_rh_src, 3)

    centroid = (weights[:, np.newaxis] * rh_src_coords[top_idx]).sum(axis=0)

    tree = cKDTree(rh_src_coords)
    _, nearest = tree.query(centroid, k=1)
    return ("rh", int(rh_src_verts[nearest]))


def _get_com_vertex_fulltime(stc, subjects_dir, com_top_num):
    """Weighted COM of the top-N rh source vertices aggregated over the full epoch.

    Uses abs_sum over all time points (same method as compute_com_over_time in
    analyze_src_loc.py with method='abs_sum'), but restricted to rh only so the
    returned FS vertex number is unambiguous.

    Returns (hemi_str, fs_vertex_number) or None.
    """
    # abs_sum of |activation| over all time points → per-vertex weight
    rh_agg = np.sum(np.abs(stc.rh_data), axis=1)
    if rh_agg.max() <= 0:
        return None

    rh_src_verts = stc.vertices[1]
    n = min(com_top_num, len(rh_agg))
    top_idx = np.argsort(rh_agg)[-n:]
    weights = rh_agg[top_idx]
    weights = weights / weights.sum()

    surf_path = os.path.join(subjects_dir, stc.subject, "surf", "rh.mid")
    rh_coords, _ = read_surface(surf_path)
    rh_src_coords = rh_coords[rh_src_verts]

    centroid = (weights[:, np.newaxis] * rh_src_coords[top_idx]).sum(axis=0)

    tree = cKDTree(rh_src_coords)
    _, nearest = tree.query(centroid, k=1)
    return ("rh", int(rh_src_verts[nearest]))


def _get_marker_vertex(stc, t_idx, subjects_dir, marker, com_top_num):
    """Return (hemi, fs_vertno) for the chosen marker strategy at a single time point, or None."""
    if marker == "peak":
        rh_data_t = stc.rh_data[:, t_idx]
        if rh_data_t.max() > 0:
            return ("rh", int(stc.vertices[1][np.argmax(rh_data_t)]))
        return None
    if marker == "com":
        return _get_com_vertex(stc, t_idx, subjects_dir, com_top_num)
    return None  # marker=None → no foci


def _get_video_marker_vertex(stc, subjects_dir, marker, com_top_num):
    """Return (hemi, fs_vertno) for the video static marker (computed over the full epoch).

    marker="peak" uses MNE's get_peak over the full epoch (positive values only).
    marker="com"  uses abs_sum COM over the full epoch via _get_com_vertex_fulltime.
    """
    if marker == "peak":
        try:
            vertno, _ = stc.get_peak(hemi="rh", mode="pos",
                                     tmin=stc.tmin, tmax=stc.times[-1],
                                     vert_as_index=False)
            return ("rh", int(vertno))
        except Exception:
            return None
    if marker == "com":
        return _get_com_vertex_fulltime(stc, subjects_dir, com_top_num)
    return None


def _add_overlays(brain, stc, t_idx, visual_labels,
                  subjects_dir=None, marker="peak", com_top_num=20, foci_scale=0.6):
    """Add focus marker and anatomical label borders at a given time index."""
    result = _get_marker_vertex(stc, t_idx, subjects_dir, marker, com_top_num)
    if result is not None:
        hemi, vertno = result
        brain.add_foci(vertno, coords_as_verts=True, hemi=hemi,
                       color="limegreen", scale_factor=foci_scale, alpha=0.8)
    for vl, color in visual_labels:
        brain.add_label(vl, borders=True, color=color)


def _render_snapshots(stc, snap_info, label, base_freq, inverse_method,
                       subjects_dir, hemi, views, visual_labels, save_dir, brain_cfg,
                       marker, com_top_num, foci_scale):
    for snap_t in snap_info["times"]:
        t_idx    = np.argmin(np.abs(stc.times - snap_t))
        t_actual = stc.times[t_idx]

        brain = _make_brain(stc, subjects_dir, hemi, views, t_actual,
                             time_viewer=False, brain_cfg=brain_cfg)
        _add_overlays(brain, stc, t_idx, visual_labels,
                      subjects_dir=subjects_dir, marker=marker,
                      com_top_num=com_top_num, foci_scale=foci_scale)

        t_ms = t_actual * 1000
        brain.add_text(0.1, 0.9,
                       f"{inverse_method}: {label} | {base_freq} Hz | {t_ms:.1f} ms",
                       "title", font_size=8)

        img_path = os.path.join(save_dir, f"{inverse_method}_{label}_{t_ms:.1f}ms.png")
        brain.save_image(img_path)
        print(f"Saved snapshot → {img_path}")
        brain.close()


def _render_video(stc, label, base_freq, inverse_method, subjects_dir, hemi, views,
                   visual_labels, save_dir, brain_cfg, marker, com_top_num, foci_scale):
    """Save a 0.5-second video at 0.2x real-time.

    time_dilation=5 stretches each real second to 5 video seconds:
    0.5s of data -> 2.5s of video (5x slower = 0.2x real-time speed).
    The foci marker is static (computed once over the full epoch) and persists
    through all frames.
    """
    t_start = stc.tmin
    t_end   = min(stc.tmin + 0.5, stc.times[-1])

    brain = _make_brain(stc, subjects_dir, hemi, views, t_start,
                         time_viewer=False, brain_cfg=brain_cfg)

    # Static marker: peak or COM over the full epoch (not just the 0.5s clip)
    marker_result = _get_video_marker_vertex(stc, subjects_dir, marker, com_top_num)
    if marker_result is not None:
        m_hemi, m_vertno = marker_result
        brain.add_foci(m_vertno, coords_as_verts=True, hemi=m_hemi,
                       color="limegreen", scale_factor=foci_scale, alpha=0.8)

    for vl, color in visual_labels:
        brain.add_label(vl, borders=True, color=color)
    brain.add_text(0.1, 0.9, f"{inverse_method}: {label} | {base_freq} Hz",
                   "title", font_size=8)

    vid_path = os.path.join(save_dir, f"{inverse_method}_{label}_video.mov")
    brain.save_movie(vid_path, tmin=t_start, tmax=t_end,
                     time_dilation=5.0, framerate=24)
    print(f"Saved video → {vid_path}")
    brain.close()


def _open_interactive(stc, subjects_dir, hemi, views, visual_labels, brain_cfg):
    """Open an interactive Brain window (no saving)."""
    brain = _make_brain(stc, subjects_dir, hemi, views, stc.tmin,
                         time_viewer=True, brain_cfg=brain_cfg)
    for vl, color in visual_labels:
        brain.add_label(vl, borders=True, color=color)
