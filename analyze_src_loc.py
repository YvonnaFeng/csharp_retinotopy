import numpy as np
from pathlib import Path
from mne.viz._brain.surface import read_surface
from scipy.spatial import cKDTree
import mne
import neuropythy as ny

# {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO1', 6: 'VO2', 7: 'LO1', 8: 'LO2', 9: 'TO1', 10: 'TO2', 11: 'V3b', 12: 'V3a'}
AREA_NAME_TO_ID = {
    'v1': 1, 'v2': 2, 'v3': 3, 'hv4': 4, 'vo1': 5, 'vo2': 6,
    'lo1': 7, 'lo2': 8, 'to1': 9, 'to2': 10, 'v3b': 11, 'v3a': 12
}

# Helper to convert Neuropythy indices -> FreeSurfer vertex IDs ---
def _to_fs_verts(retino_hemi, idx_array):
    # Many ny objects expose .vertex (FS IDs). If not, assume idx are already FS IDs.
    if idx_array is None:
        return np.array([], dtype=int)
    if hasattr(retino_hemi, "vertex"):
        return np.asarray(retino_hemi.vertex[idx_array], dtype=int)
    return np.asarray(idx_array, dtype=int)

def _normalize_area_to_ids(area):
    """
    area: str like 'v1' / 'all', or iterable like ['v1','v2','v3'].
    Returns: set of ints (area IDs) or None for 'all'.
    """
    if area is None:
        return {AREA_NAME_TO_ID['v1']}  # default to V1 if None given (keep your old default)
    if isinstance(area, str):
        key = area.strip().lower()
        if key == 'all':
            return None
        if key not in AREA_NAME_TO_ID:
            valid = ', '.join(['all'] + list(AREA_NAME_TO_ID.keys()))
            raise ValueError(f"Unknown area '{area}'. Use one of: {valid}")
        return {AREA_NAME_TO_ID[key]}
    # list/tuple of names
    ids = set()
    for a in area:
        key = str(a).strip().lower()
        if key == 'all':
            return None  # 'all' overrides others
        if key not in AREA_NAME_TO_ID:
            valid = ', '.join(['all'] + list(AREA_NAME_TO_ID.keys()))
            raise ValueError(f"Unknown area '{a}'. Use one of: {valid}")
        ids.add(AREA_NAME_TO_ID[key])
    return ids

def _weights_one_hemi(retino, e0, th0, sigma_stim, allowed_ids):
    """
    Compute soft Gaussian weights; restrict to allowed_ids (set[int]) or no restriction if None.
    """
    th = np.deg2rad(retino.angle)
    x  = retino.eccen * np.cos(th)
    y  = retino.eccen * np.sin(th)
    x0 = e0 * np.cos(th0)
    y0 = e0 * np.sin(th0)

    if hasattr(retino, 'sigma') and (retino.sigma is not None):
        sigma_eff = np.sqrt((sigma_stim ** 2) + (retino.sigma ** 2))
    else:
        a, b = 0.38, 0.15
        sigma_eff = np.sqrt((sigma_stim ** 2) + (a + b * retino.eccen) ** 2)
    sigma_eff = np.maximum(sigma_eff, 1e-6)

    dx = x - x0
    dy = y - y0
    w  = np.exp(-0.5 * (dx*dx + dy*dy) / (sigma_eff * sigma_eff))

    if allowed_ids is not None:  # mask to selected areas
        mask = np.isin(retino.varea, list(allowed_ids))
        w = w * mask.astype(float)

    keep = w > 1e-4
    idx  = np.where(keep)[0]
    w    = w[keep]

    if w.size > 0:
        w = w / np.max(w)  # optional normalization to [0,1] per hemi selection
    return idx, w

def choose_roi_weighted(roi, lh_retino, rh_retino, only_left=False, only_right=False, area='v1'):
    """
    area: str ('v1', ..., 'all') or list/tuple of areas (['v1','v2','v3']).
    """
    stim = {
        'up':           {'eccen': 5.0,  'angle': 45.0,  'size': 2.4},
        'low':          {'eccen': 5.0,  'angle': 135.0, 'size': 2.4},
        'center-small': {'eccen': 1.25, 'angle': 90.0,  'size': 0.83},
        'center-large': {'eccen': 2.0,  'angle': 90.0,  'size': 1.14},
    }[roi]

    e0  = float(stim['eccen'])
    th0 = np.deg2rad(float(stim['angle']))
    S   = float(stim['size'])
    sigma_stim = S / 2.355  # FWHM→sigma

    allowed_ids = _normalize_area_to_ids(area)

    v_left,  w_left  = _weights_one_hemi(lh_retino, e0, th0, sigma_stim, allowed_ids)
    v_right, w_right = _weights_one_hemi(rh_retino, e0, th0, sigma_stim, allowed_ids)

    # default to RH if neither specified (keep your old behavior)
    if not (only_left or only_right):
        only_left = False
        only_right = True

    if only_left:
        print(f"Number of original vertices in {roi} (LH): {len(v_left)}")
        return {'lh_idx': v_left, 'lh_w': w_left}
    elif only_right:
        print(f"Number of original vertices in {roi} (RH): {len(v_right)}")
        return {'rh_idx': v_right, 'rh_w': w_right}
    else:
        return {'lh_idx': v_left, 'lh_w': w_left, 'rh_idx': v_right, 'rh_w': w_right}

def get_peak_idx(subject, stc, top_num=5, time_idx = 0, constraint_idx=None):
    # Extract Peak Source Localization
    stc_data = np.abs(stc.data[:, time_idx])  # Take absolute values

    # Normalize the activity values
    stc_data /= stc_data.sum()

    if constraint_idx is None:
        top_idx = np.argsort(stc_data)[-top_num:]  # Indices of the top 5 active sources
    else: # should be in roi_verts
        lh_src_verts = stc.vertices[0]
        rh_src_verts = stc.vertices[1]
        src_verts = np.concatenate([lh_src_verts, rh_src_verts])
        # Build a lookup: map vertex number to stc index
        vert_to_idx = {v: i for i, v in enumerate(src_verts)}

        # Filter only vertices within the constraint that exist in the source space
        constraint_indices = [vert_to_idx[v] for v in constraint_idx if v in vert_to_idx]

        # Get activity values within ROI
        constrained_data = stc_data[constraint_indices]

        # Sort and get top N indices
        top_idx_within_roi = np.argsort(constrained_data)[-top_num:]

        # Map back to stc.data full index
        top_idx = np.array(constraint_indices)[top_idx_within_roi]

    # Scale the Activity Values (Normalize & Multiply)
    scaled_values = stc_data[top_idx] / np.max(stc_data[top_idx])  # Normalize between 0 and 1
    scaled_values *= 10  # Multiply by a factor (adjust as needed)

    new_data = np.zeros_like(stc.data)  # Initialize all zeros
    new_data[top_idx, time_idx] = scaled_values  # Assign scaled activity values

    stc_peak = mne.SourceEstimate(
        data=new_data,
        vertices=stc.vertices,
        tmin=stc.tmin,
        tstep=stc.tstep,
        subject=subject
    )
    return top_idx, stc_peak

def compute_peak_com(subjects_dir, subject, fwd, stc, time_idx=125, top_num=5, constraint_idx=None, n_neighbor=5):
    lh_coords, _ = read_surface(f"{subjects_dir}/{subject}/surf/lh.mid")
    rh_coords, _ = read_surface(f"{subjects_dir}/{subject}/surf/rh.mid")
    # Get coordinates for vertices used in this stc
    lh_src_coords = lh_coords[stc.vertices[0]]
    rh_src_coords = rh_coords[stc.vertices[1]]
    src_pos = np.vstack([lh_src_coords, rh_src_coords])

    # Select activity values and vertex positions at the peak indices
    peak_indices, _ = get_peak_idx(subject, stc, top_num=top_num, time_idx=time_idx, constraint_idx=constraint_idx)
    peak_coords = src_pos[peak_indices]
    activity_vals = np.abs(stc.data[peak_indices, time_idx]) # get the abs amplitude of peaks
    activity_vals /= activity_vals.sum()  # Normalize to use as weights which sum to 1

    # Compute weighted center of mass (centroid)
    # multiply vertex coor (k,3) with weight (k,1) -> weighted average position (k,3)
    # sum over all vertices (k,3) -> (3,) weighted average (centroid) in 3d
    centroid_pos = np.sum(peak_coords * activity_vals[:, np.newaxis], axis=0)

    # Check how close centroid is to actual vertices
    tree = cKDTree(src_pos) #spatial index for nearest neighbor lookup
    dists, nearest_indices = tree.query(centroid_pos, k=n_neighbor) #k nearest vertex coor to centroid_pos
    nearest_indices = np.atleast_1d(nearest_indices)
    nearest_vertex_coords = src_pos[nearest_indices]

    return centroid_pos, nearest_vertex_coords, nearest_indices

def get_roi_centroid(subjects_dir, subject, fwd, stc, roi, left=False, right=True, area=["v1", "v2", "v3"]):
    retinotopy = ny.freesurfer_subject(subject)
    (lh_retino, rh_retino) = ny.vision.predict_retinotopy(retinotopy)
    out = choose_roi_weighted(roi, lh_retino, rh_retino,
        only_left=left, only_right=right, area=area)
    
    lh_fs = np.array([], dtype=int)
    lh_w  = np.array([], dtype=float)
    rh_fs = np.array([], dtype=int)
    rh_w  = np.array([], dtype=float)
    if ('lh_idx' in out) and ('lh_w' in out) and (left or (not left and not right)):
        lh_fs = _to_fs_verts(lh_retino, np.asarray(out['lh_idx']))
        lh_w  = np.asarray(out['lh_w'], dtype=float)
    if ('rh_idx' in out) and ('rh_w' in out) and (right or (not left and not right)):
        rh_fs = _to_fs_verts(rh_retino, np.asarray(out['rh_idx']))
        rh_w  = np.asarray(out['rh_w'], dtype=float)

    lh_src_verts = np.asarray(stc.vertices[0], dtype=int)
    rh_src_verts = np.asarray(stc.vertices[1], dtype=int)
    if lh_fs.size:
        mask_lh_in_src = np.isin(lh_fs, lh_src_verts)
        lh_fs = lh_fs[mask_lh_in_src]
        lh_w  = lh_w[mask_lh_in_src]
    if rh_fs.size:
        mask_rh_in_src = np.isin(rh_fs, rh_src_verts)
        rh_fs = rh_fs[mask_rh_in_src]
        rh_w  = rh_w[mask_rh_in_src]
    # If no vertices survive, bail early
    if (lh_fs.size == 0) and (rh_fs.size == 0):
        return None

    # Load RH surface geometry (white matter) 
    rh_coords, _ = read_surface(f"{subjects_dir}/{subject}/surf/rh.mid")
    lh_coords, _ = read_surface(f"{subjects_dir}/{subject}/surf/lh.mid")

    coords_list = []
    weights_list = []
    if lh_fs.size:
        coords_list.append(lh_coords[lh_fs])
        weights_list.append(lh_w)
    if rh_fs.size:
        coords_list.append(rh_coords[rh_fs])
        weights_list.append(rh_w)
    coords = np.vstack(coords_list)
    weights = np.concatenate(weights_list)

    # Compute centroid (weighted; fallback to unweighted if all-zero)
    wsum = weights.sum()
    if np.isfinite(wsum) and (wsum > 0):
        centroid = (weights[:, None] * coords).sum(axis=0) / wsum
    else:
        centroid = coords.mean(axis=0)

    return centroid

def compute_com_over_time(
    subjects_dir, subject, stc,
    method="abs_sum",          # 'abs_sum' (|x|), 'power' (x^2), or 'abs_max'
    time_slice=None,           # (tmin_idx, tmax_idx) inclusive; None = all time
    constraint_idx=None,       # list/array of FS vertex IDs to restrict to (both hemi mixed OK)
    top_num=None,              # optionally keep only top-N vertices by aggregated weight
    n_neighbor=5               # return nearest vertices around COM
):
    """
    centroid_pos : (3,) np.ndarray
    nearest_vertex_coords : (k, 3) np.ndarray
    nearest_indices : (k,) np.ndarray indices into the stacked src vertex array
    """

    # Select time window
    data = stc.data  # shape: (n_vertices_total, n_times)
    if time_slice is None:
        t0, t1 = 0, data.shape[1]
    else:
        t0, t1 = time_slice
        t0 = int(t0)
        t1 = int(t1) + 1  # make inclusive
        t0 = max(0, t0); t1 = min(data.shape[1], t1)
    data_win = data[:, t0:t1]

    # Aggregate over time -> per-vertex weights
    if method == "abs_sum":
        vert_w = np.sum(np.abs(data_win), axis=1)
    elif method == "power":
        vert_w = np.sum(data_win**2, axis=1)
    elif method == "abs_max":
        vert_w = np.max(np.abs(data_win), axis=1)
    else:
        raise ValueError("method must be one of {'abs_sum','power','abs_max'}")

    # Build coordinates for vertices used in this STC (FS vertex IDs)
    lh_coords, _ = read_surface(str(Path(subjects_dir) / subject / "surf" / "lh.mid"))
    rh_coords, _ = read_surface(str(Path(subjects_dir) / subject / "surf" / "rh.mid"))

    lh_src_verts = np.asarray(stc.vertices[0], dtype=int)
    rh_src_verts = np.asarray(stc.vertices[1], dtype=int)

    lh_src_coords = lh_coords[lh_src_verts]
    rh_src_coords = rh_coords[rh_src_verts]
    src_pos = np.vstack([lh_src_coords, rh_src_coords])  # (n_vertices_total, 3)

    # restrict to an ROI by FS vertex IDs (both hemi OK)
    if constraint_idx is not None and len(constraint_idx) > 0:
        constraint_idx = np.asarray(constraint_idx, dtype=int)
        # Map FS vert -> index in stacked (lh then rh)
        src_verts_all = np.concatenate([lh_src_verts, rh_src_verts])
        vert_to_idx = {v: i for i, v in enumerate(src_verts_all)}
        keep = [vert_to_idx[v] for v in constraint_idx if v in vert_to_idx]
        if len(keep) == 0:
            raise ValueError("None of the constraint_idx vertices are in the STC source space.")
        keep = np.asarray(keep, dtype=int)
        mask = np.zeros_like(vert_w, dtype=bool)
        mask[keep] = True
        vert_w = vert_w[mask]
        src_pos = src_pos[mask]

    # keep only top-N vertices by aggregated weight
    if top_num is not None and top_num > 0 and top_num < len(vert_w):
        top_idx = np.argsort(vert_w)[-top_num:]
        vert_w = vert_w[top_idx]
        src_pos = src_pos[top_idx]

    # Normalize weights (avoid division by zero)
    wsum = float(vert_w.sum())
    if wsum <= 0:
        # fallback: unweighted centroid over the selected vertices
        centroid_pos = src_pos.mean(axis=0)
    else:
        w = vert_w / wsum
        centroid_pos = (w[:, None] * src_pos).sum(axis=0)

    # Nearest vertices to the centroid (for reporting/rounding to mesh)
    tree = cKDTree(src_pos)
    dists, nearest_indices = tree.query(centroid_pos, k=n_neighbor)
    nearest_indices = np.atleast_1d(nearest_indices)
    nearest_vertex_coords = src_pos[nearest_indices]

    return centroid_pos, nearest_vertex_coords, nearest_indices

def compute_loc_error(subjects_dir, subject, fwd, stc, roi, left=False, right=True, area=["v1", "v2", "v3"],top_num=32, n_neighbor=5, constraint_idx=None):
    inv_centroid, _, inv_centroid_idx = compute_com_over_time(subjects_dir, subject, stc, method="abs_sum",time_slice=None, constraint_idx=constraint_idx,top_num=top_num,n_neighbor=n_neighbor)
    # compute_peak_com(subjects_dir, subject, fwd, stc, time_idx=time_idx, top_num=top_num, n_neighbor=n_neighbor, constraint_idx=constraint_idx)
    true_roi_centroid = get_roi_centroid(subjects_dir, subject, fwd, stc, roi, left, right, area)

    # Euclidean distance in meters
    distance = np.linalg.norm(inv_centroid - true_roi_centroid)
    # print(f"idx from function: {inv_centroid_idx}")
    # print(f"location from function: {inv_centroid}")
    print(f"Distance between estimated CoM and ROI: {distance:.2f} mm")
    return distance