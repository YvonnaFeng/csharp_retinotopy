#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 23 10:11:54 2026

@authors: Qiyuan Feng, Adapted from MEG_pipeline test from Alexandria McPherson and Qiyuan Feng

C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

**OPM-MEG: Do coregistration through YORC script FIRST to generate "trans" object
Full pipeline description: 
    1. Load data (CTF vs FIF files)
    2. Preprocessing - Reject eyeblinks with ICA, high-pass filter, add functions for SSS, homogenous field correction, etc
    3. Make evokeds for each condition
    4. Create covariance
    5. Forward solution
    6. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
    7. Apply inverse to evokeds
    8. Put into BIDS structure, specifically with events
    9. Generate MNE-report

"""
# --- Dependencies ------------------------------------------------------------
import os
import pandas as pd
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_ctf
from mne import Covariance
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator, apply_inverse)
from mne.beamformer import make_lcmv, apply_lcmv
from mne.surface import read_surface
# This takes care some numpy dependency issues...not required depending on the numpy version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import meg_load_preprocess as meg
from make_forward_inverse import make_forward, make_inverse, get_filtered_stc


def plot_fft_histogram_meg(epochs, picks='mag', fmax=80,
                           target_freqs=[10, 12, 15, 20, 24, 30, 36, 40, 45, 48, 60],
                           title='FFT Amplitude Histogram'):
    """Average-then-amplitude FFT across epochs and channels. Units: fT for mag."""
    data = epochs.get_data(picks=picks)   # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    freqs = np.fft.rfftfreq(data.shape[-1], d=1. / sfreq)

    complex_fft = np.fft.rfft(data, axis=-1)
    mean_amp = np.abs(complex_fft.mean(axis=(0, 1))) * 1e15   # T → fT

    mask = freqs <= fmax
    plot_freqs, plot_amp = freqs[mask], mean_amp[mask]
    freq_res = freqs[1] - freqs[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(plot_freqs, plot_amp, width=freq_res * 0.9,
           color='steelblue', alpha=0.7, align='center')
    for f in (target_freqs or []):
        ax.axvline(f, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.set_xticks(np.arange(0, fmax + 5, 5))
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (fT)',
           title=title, xlim=[-freq_res / 2, fmax + freq_res])
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# --- Main (example usage) ----------------------------------------------------
if __name__ == '__main__':
   # --- Load user-specific config (copy config_template.py -> config.py and fill in your paths)
    from config import sample_dir_meg, raw_files_meg, trans_meg, subjects_dir, subject, viz_bool, sss_bool, save_report, save_dir

    sample_dir = sample_dir_meg
    raw_files = raw_files_meg
    trans = trans_meg
    
    ## 1. Load and setup data
    for file in raw_files_meg:
        # --- 1. Load data, find events ---------------------------------------
        if file.endswith(".fif"):
            ## load OPM, find events, do preprocessing
            ## specify trigger 
            trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
            
            #setup raw, info, events, and specify task type
            raw = mne.io.read_raw_fif(os.path.join(sample_dir,file),'default', preload=True)
            [events_df,events,task] = meg.get_events_fif(raw,file,trigger_chan)
            info = raw.info
            picks = 'mag'
            reject_criteria = dict(mag=4000e-15)  # 4000fT

        elif file.endswith(".ds"):
            raw = read_raw_ctf(os.path.join(sample_dir,file), preload=True)
            [events_df,events,task] = meg.get_events_ctf(raw,file)
            # always do this preprocessing, reccommended by Dylan @ UCSF
            raw.apply_gradient_compensation(3)
            info = raw.info
            picks = 'grad'
            reject_criteria = dict(grad=4000e-13) #4000 fT/cm
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
        
        
        # --- 1.B Look at Events -----------------------------------------------
        ## save events
        # mne.write_events( participant + '/' + participant + '_events.fif',events)
        sfreq = raw.info["sfreq"]
        if viz_bool:
            fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)
        ##set up report
        if save_report:
            report = mne.Report(title="Report for subject: "+subject + ", Task: "+ task)
            report.add_raw(raw=raw, title= file , psd=True)
            report.add_trans(trans=trans, info=raw.info, title='Coregistration',subject=subject,subjects_dir=subjects_dir)
            report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
            

        # --- 2. A Preprocess -------------------------------------------------
        ## start with methods common to both CTF and OPM-MEG
        #-- remove bad channels, check for NaNs
        bads = raw.info["bads"]
        raw.drop_channels(bads)
        bads_NaNs=[]
        for i in range(0,raw.info["nchan"]):
            ch_pos = raw.info["chs"][i]["loc"][:3]
            if np.isnan(ch_pos).any():
                bads_NaNs.append(raw.info["chs"][i]["ch_name"])
        raw.drop_channels(bads_NaNs)
        
        #-- Notch filter 60Hz, low pass and high pass
        freq_min = 0.5
        freq_max = 50
        raw = meg.filter_raw(raw,freq_min,freq_max)
        #downsample?
        
        #-- Do SSS
        if sss_bool:
            Lin=8
            raw = meg.sss_prepros(raw,Lin)
        
        #-- do SSP, one projector
        #raw_pre = ssp_filter(raw)
        
        
        # --- 2.B visualize sensor alignment and BEM---------------------------------
        if viz_bool:
            mne.viz.plot_alignment(
                info,
                trans=trans,
                subject=subject,
                dig=True,
                meg=["helmet", "sensors"],
                subjects_dir=subjects_dir,
                surfaces="head-dense",
                )
            # look at BEM
            plot_bem_kwargs = dict(
                subject=subject,
                subjects_dir=subjects_dir,
                brain_surfaces="white",
                orientation="coronal",
                slices=[50, 100, 150, 200])
            mne.viz.plot_bem(**plot_bem_kwargs)
            
        # --- 3. Make Epochs and Evokeds --------------------------------------
        # Mirror EEG epoch structure: 5 core bins + prelude + postlude
        tmin = 0.0
        tmax = 2.0
        sfreq = raw.info['sfreq']
        n_samples  = int(round((tmax - tmin) * sfreq))
        tmax_exact = tmin + (n_samples - 1) / sfreq

        event_id = {
            'bin/0': 0, 'bin/1': 1, 'bin/2': 2, 'bin/3': 3, 'bin/4': 4,
            'noise/prelude':  5,
            'noise/postlude': 6,
        }

        epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax_exact,
                    baseline=None,
                    reject=None,
                    preload=True)

        evokeds = {
            'bin':   epochs['bin'].average(),   # all 5 core bins averaged together
            'noise': epochs['noise'].average(),  # prelude + postlude averaged together
        }
        if save_report:
            report.add_evokeds(evokeds=evokeds['bin'], titles=['bin'])

            # Sensor layout — use this to identify occipital channels by position
            fig_sensors = epochs.plot_sensors(show_names=True, show=False)
            report.add_figure(fig_sensors, title='Sensor layout (OPM helmet)',
                              section='Epochs QC')
            plt.close(fig_sensors)

            # FFT — all MEG channels averaged together
            fig_fft_all, _ = plot_fft_histogram_meg(
                epochs['bin'], picks='mag',
                title='FFT — all MEG channels (core bins)')
            report.add_figure(fig_fft_all, title='FFT — all channels',
                              section='Epochs QC')
            plt.close(fig_fft_all)
        if viz_bool:
            ts_args = dict(time_unit="s")
            topomap_args = dict(time_unit="s")
            fig = evokeds['bin'].plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args, title=file + ' Task: ' + task + ', Condition: bin')
        
        # --- 4. Create covariance --------------------------------------------
        cov = mne.compute_covariance(epochs['noise'], tmax=None, projs=None, method="empirical", rank=None)
        cov = mne.cov.regularize(cov, epochs['noise'].info, mag=0.05, grad = 0.05, proj = True, exclude = 'bads')
        cov.save(f'{sample_dir}/V1Loc_empirical_cov.fif', overwrite=True)

        # --- 5. Forward solution ---------------------------------------------
        fwd = make_forward(subject, subjects_dir, trans, evokeds['bin'],
                         overwrite_fwd=False, overwrite=False,
                         fixed=True, bem_ico=4, src_space="oct7",
                         conductivity=(0.3, 0.006, 0.3),
                         mindist=5, surface='mid',
                         visualize=False, verbose=False, eeg=False, meg=True)
        
        # --- 6. Inverse solution ---------------------------------------------
        # fwd = mne.read_forward_solution(f'{subjects_dir}/{subject}/bem/{subject}-fwd.fif')
        # cov = mne.read_cov(f'{sample_dir}/V1Loc_empirical_cov.fif')
        # # Restrict forward to channels present in the evoked (handles bad-channel drops)
        # fwd = mne.pick_channels_forward(fwd, evokeds['bin'].ch_names, ordered=False)
        inverse_method ='dSPM'
        stc, inv_op = make_inverse(subjects_dir, subject, fwd, evokeds['bin'], cov, inverse_method=inverse_method)

        filtered_stcs = get_filtered_stc(fwd, epochs['bin'], cov,
                     filters=((5.0, "even", "center"),
                              (6.0, "even", "upper"),
                              (7.5, "even", "lower")),
                     max_harmonic_order=4,
                     fixed_ori=True,
                     snr=3, lambda2=None,
                     inverse_method=inverse_method,
                     save_dir=None, eeg=False, meg=True)
        
        # --- 8. Visualize inverse --------------------------------------------
        if viz_bool:
            stc_plot = stc.copy().crop(tmin=stc.tmin, tmax=0.5)
            vertno_max, time_max = stc_plot.get_peak(hemi="rh", tmin=stc.tmin, tmax=0.5)
            surfer_kwargs = dict(
                hemi="split",
                subjects_dir=subjects_dir, # clim=dict(kind="value"), lims=[12,20,28]
                views=["caudal", "medial"], # for visual task
                initial_time=time_max,
                time_unit="s",
                size=(800, 800),
                smoothing_steps=5)

            brain = stc_plot.plot(**surfer_kwargs)
            brain.plotter.scalar_bar.GetLabelTextProperty().SetFontSize(8)
            # These params are tested to fit if you have two views (one on top and one at the bottom) with splitted hemi. eg. views=["caudal", "medial"]
            # You should change this if you have single view only
            sb = brain.plotter.scalar_bar
            x, _ = sb.GetPosition()
            sb.SetPosition(x, 0.6)  # vertically between caudal (top) and medial (bottom) rows
            w, h = sb.GetPosition2()
            sb.SetPosition2(w, h * 0.5)  # shrink height so top of the color bar doesn't clip
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
            brain.add_text(0.1, 0.9, "dSPM, Task: " + task, "title", font_size=8)

            video_path = os.path.join(save_dir, f"{inverse_method}_all.mp4")
            brain.save_movie(video_path, time_dilation=5, framerate=24)
            print(f"Saved video → {video_path}")
            brain.close()

            # filtered stc
            for label, result in filtered_stcs.items():
                stc = result["stc"]
                stc_plot = stc.copy().crop(tmin=stc.tmin, tmax=0.5)
                vertno_max, time_max = stc_plot.get_peak(hemi="rh", tmin=stc.tmin, tmax=0.5)

                surfer_kwargs = dict(
                    hemi="split",
                    subjects_dir=subjects_dir,
                    views=["caudal", "medial"],
                    initial_time=time_max,
                    time_unit="s",
                    size=(800, 800),
                    smoothing_steps=5)

                brain = stc_plot.plot(**surfer_kwargs)

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
                brain.save_movie(video_path, time_dilation=5, framerate=24)
                print(f"Saved video → {video_path}")
                brain.close()
        
        # --- 9. Generate and save MNE Report ---------------------------------
        if save_report:
            report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
            report.add_covariance(cov=cov, info=raw.info, title="Covariance")
            report.add_bem(subject=subject, title='BEM')
            report.add_stc(stc=stc, subject=subject, subjects_dir=subjects_dir, title="STC")
            report.save(f"{save_dir}/report_raw.html", overwrite=True)
            print(f"Report saved to {save_dir}/report_raw.html")
    



