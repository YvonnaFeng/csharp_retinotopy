#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:40:01 2026

* This version is adapted from the original MEG_load_preprocess

@author: Alexandria McPherson


C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

Full pipeline description: 
    1. Load data (CTF vs FIF files) - make functions for each data type
    2. Do coregistration, either in line or through YORC script
    3. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
    4. Make evokeds for each condition
    5. Create covariance
    6. Forward solution
    7. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
    8. Apply inverse to evokeds
    9. Put into BIDS structure, specifically with events
    10. Generate MNE-report

This module covers steps 1-4 of the full pipeline:
    1. Load data (CTF vs FIF files) - make functions for each data type
    2. Do coregistration, either in line or through YORC script
    3. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
    4. Make evokeds for each condition - parsed with event IDs
"""

# --- Dependencies -----------------------------------------------------------
import mne
import os
from mne.io import read_raw_ctf
from mne.minimum_norm import apply_inverse

import numpy as np
import pandas as pd
from pathlib import Path


# --- Helpers -----------------------------------------------------------------
def _highpass_filter_opm(raw):
    """ 'high-filter' = applies high-pass filter 3Hz ('Agressive'), low pass 40Hz, 60Hz notch """
    # apply high-pass filter 
    freq_min = 3
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # always notch filter
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def _ssp_filter(raw):
    """ 'ssp-filter' = applies high-pass filter 2Hz, low pass 40Hz, 60Hz notch, and SSP method """
    # high pass
    freq_min = 2
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # always notch filter
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=1, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def _sss_prepros(raw):
    """ 'sss-filter' = applies high-pass filter 1Hz, low pass 40Hz, 60Hz notch, and SSS method """
    # freq_min = 1
    # freq_max = 40       
    # # # apply high-pass filter 
    # raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # # # apply notch filter for 60Hz power lines
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=1.0, extended_proj=(), verbose=None)  
    freq_min = 2
    freq_max = 40       
    # # apply high-pass filter 
    raw_sss.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    return raw_sss

def pros_OPM_data(raw, trigger_chan, prepros_type):
    """Load OPM raw data, find events on trigger chan, do specified preprocessing 

    Parameters
    ----------
    raw: mne.Raw
    trigger chan: str
        name of the trigger channel
    prepros_type: str
        pick from the following options 
        'high-filter' = applies high-pass filter 3Hz, low pass 40Hz, 60Hz notch filter.
        'ssp-filter' = applies high-pass filter 2Hz, low pass 40Hz, 60Hz notch, and SSP proj from baseline
        'sss-filter' = applies high-pass filter 1Hz, low pass 40Hz, 60Hz notch, and SSS method
        TODO: add Fosters Inverse with SSS
        
    Returns
    -------
    rawp : mne.Raw with applied filters/projectors
    events: arrary (event time x 3) containing event IDs and onsets
    """
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    ## preprocess
    if prepros_type == 'high-filter':
        rawp = _highpass_filter_opm(raw)
    elif prepros_type == 'ssp-filter':
        rawp = _ssp_filter(raw)
    elif prepros_type == 'sss-filter':
        rawp = _sss_prepros(raw)
    else:
        print("please pick preprocessing type from the defined options")
    return rawp,events

# --- Load, find events, and rename/order them --------------------------------
def get_events_fif(raw,file, trigger_chan):
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    if "VWFA" in file:
        task = "VWFA"
        event_code_list = events[:, 2]
        event_code_updates = np.zeros_like(event_code_list)
        special_codes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ei = 0
        while ei < len(event_code_list):
            event = event_code_list[ei]
            if event in special_codes:
                event_code_updates[ei+1:ei+5] = event
                event_code_updates[ei] = 201  # code for what was the condition label
                ei += 4  # skip next 4 positions 
            else:
                ei += 1  # just advance by 1 if no match
        events[:, 2] = event_code_updates
        # get just the code part
        trigger_codes = events[:, 2]
        blocks = trigger_codes.reshape(-1, 5)
        blocks_rearranged = blocks[:, [1, 2, 3, 4, 0]]
        result = blocks_rearranged.flatten()
        events[:, 2] = result
        # make event codes interpretable
        code_dict = {1: 'highFreqWords_Sloan',
                     2: 'pseudowords_Sloan',
                     3: 'consonants_Sloan',
                     4: 'falseFontsHigh_Sloan',
                     5: 'highFreqWords_Courier',
                     6: 'pseudowords_Courier',
                     7: 'consonants_Courier',
                     8: 'falseFontsHigh_Courier',
                     9: 'background_',
                     201: 'stimulusoffset_'
                     }
        
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['font'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
        
    if "Tones" in file:
        task = "Tones"
        ## tones don't need modification (?)
        # event_code_list = events[:, 2]
        # event_code_updates = np.zeros_like(event_code_list)
        # special_codes = np.array([10, 11, 12, 13, 14, 15])
        # ei = 0
        # while ei < len(event_code_list):
        #     event = event_code_list[ei]
        #     if event in special_codes:
        #         event_code_updates[ei+1:ei+2] = event
        #         event_code_updates[ei] = 201  # code for what was the condition label
        #         ei += 2  # skip next 1 positions 
        #     else:
        #         ei += 1  # just advance by 1 if no match
        # events[:, 2] = event_code_updates
        # # get just the code part
        # trigger_codes = events[:, 2]
        # blocks = trigger_codes.reshape(-1, 2)
        # blocks_rearranged = blocks[:, [1, 0]]
        # result = blocks_rearranged.flatten()
        # events[:, 2] = result
        ## make event codes interpretable
        code_dict = {10: '250_Hz',
                     11: '500_Hz',
                     12: '1000_Hz',
                     13: '2000_Hz',
                     14: '4000_Hz',
                     15: 'background_',
                     200: 'stimulusoffset_'
                     }
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['units'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
        
        
        
    if "V1Loc" in file:
        task = "V1Loc"
        TRIAL_ID = 16   # trial-onset trigger (equivalent to EEG DIN4)
        BIN_ID   = 200  # bin-onset trigger   (equivalent to EEG DIN5)

        trial_samples = events[events[:, 2] == TRIAL_ID, 0]
        bin_samples   = events[events[:, 2] == BIN_ID,   0]

        # Mirror EEG epoch structure:
        #   DIN4-equiv ── prelude ── bins[0..4] (5 core) ── bin[5] postlude ── bin[6] ITI (not epoched)
        event_id = {
            'bin/0': 0, 'bin/1': 1, 'bin/2': 2, 'bin/3': 3, 'bin/4': 4,
            'noise/prelude':  5,
            'noise/postlude': 6,
        }
        label_map = {v: k for k, v in event_id.items()}

        all_events = []
        for i, trial_start in enumerate(trial_samples):
            trial_end = trial_samples[i + 1] if i + 1 < len(trial_samples) else np.inf
            bins_in_trial = bin_samples[(bin_samples >= trial_start) & (bin_samples < trial_end)]

            for pos, s in enumerate(bins_in_trial[:5]):
                all_events.append([s, 0, pos])

            if len(bins_in_trial) >= 1:
                all_events.append([trial_start, 0, 5])   # prelude anchored at trial trigger

            if len(bins_in_trial) >= 6:
                all_events.append([bins_in_trial[5], 0, 6])  # postlude anchored at bin[5]
            # bins_in_trial[6] is ITI onset — not epoched

        all_events = np.array(all_events, dtype=int)
        all_events = all_events[np.argsort(all_events[:, 0])]
        events = all_events

        events_df = pd.DataFrame({'condition': [label_map[e[2]] for e in all_events]})

    else:
        print("no valid events detected, please double check data file name")
    return events_df, events, task

# -- events for CTF
def get_events_ctf(raw,file):
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    if "VWFA" in file:
        task = "VWFA"
        event_code_list = events[:, 2]
        event_code_updates = np.zeros_like(event_code_list)
        special_codes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ei = 0
        while ei < len(event_code_list):
            event = event_code_list[ei]
            if event in special_codes:
                event_code_updates[ei+1:ei+5] = event
                event_code_updates[ei] = 13107200  # code for what was the condition label
                ei += 4  # skip next 4 positions 
            else:
                ei += 1  # just advance by 1 if no match
        events[:, 2] = event_code_updates
        # get just the code part
        trigger_codes = events[:, 2]
        blocks = trigger_codes.reshape(-1, 5)
        blocks_rearranged = blocks[:, [1, 2, 3, 4, 0]]
        result = blocks_rearranged.flatten()
        events[:, 2] = result
        # make event codes interpretable
        code_dict = {65536: 'highFreqWords_Sloan',
                     131072: 'pseudowords_Sloan',
                     196608: 'consonants_Sloan',
                     262144: 'falseFontsHigh_Sloan',
                     327680: 'highFreqWords_Courier',
                     393216: 'pseudowords_Courier',
                     458752: 'consonants_Courier',
                     524288: 'falseFontsHigh_Courier',
                     589824: 'background_',
                     13107200 : 'stimulusoffset_'
                     }
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['font'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
        
    if "Tones" in file:
        code_dict = {#10: '250_Hz', ##broken code
                     720896: '500_Hz',
                     786432: '1000_Hz',
                     851968: '2000_Hz',
                     917504: '4000_Hz',
                     983040: 'background_',
                     13107200: 'stimulusoffset_'
                     }
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['units'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
           
    if "V1Loc" in file:
        task = "V1Loc"
        TRIAL_ID = 16   # trial-onset trigger (equivalent to EEG DIN4)
        BIN_ID   = 200  # bin-onset trigger   (equivalent to EEG DIN5)

        trial_samples = events[events[:, 2] == TRIAL_ID, 0]
        bin_samples   = events[events[:, 2] == BIN_ID,   0]

        event_id = {
            'bin/0': 0, 'bin/1': 1, 'bin/2': 2, 'bin/3': 3, 'bin/4': 4,
            'noise/prelude':  5,
            'noise/postlude': 6,
        }
        label_map = {v: k for k, v in event_id.items()}

        all_events = []
        for i, trial_start in enumerate(trial_samples):
            trial_end = trial_samples[i + 1] if i + 1 < len(trial_samples) else np.inf
            bins_in_trial = bin_samples[(bin_samples >= trial_start) & (bin_samples < trial_end)]

            for pos, s in enumerate(bins_in_trial[:5]):
                all_events.append([s, 0, pos])

            if len(bins_in_trial) >= 1:
                all_events.append([trial_start, 0, 5])

            if len(bins_in_trial) >= 6:
                all_events.append([bins_in_trial[5], 0, 6])

        all_events = np.array(all_events, dtype=int)
        all_events = all_events[np.argsort(all_events[:, 0])]
        events = all_events

        events_df = pd.DataFrame({'condition': [label_map[e[2]] for e in all_events]})

    else:
        print("no valid events detected, please double check data file name")
    return events_df, events, task


## -- Preprocessing Functions -------------------------------------------------
def filter_raw(raw,freq_min,freq_max):
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def ssp_filter(raw):
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=2, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def sss_prepros(raw,Lin):
    """do traditional SSS with origin 0 in MEG frame"""
    assert raw.info["bads"] == [] # double check bads were dropped
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=Lin, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)  
    return raw_sss

def _eog_artifact(raw):
    """TODO: https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html """
    eog_event_id = 512
    eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id)
    raw.add_events(eog_events, "STI 014")
    return raw 

# ---- Helpers from Teresa Cheung --------------------------------------------
def plot3Dhelmetwithhpi(raw,ax,showLabels=True,showDevice=True,thetitle=''):

    raw.load_data()

    ax.title.set_text(thetitle)
    trans=raw.info['dev_head_t']['trans']
    print(trans)

    info=raw.info
    digpts=np.array([],dtype=float)
    digpts_head=np.array([],dtype=float)

    offset=0
    picks = mne.pick_types(info, meg='mag')
    for j in picks:
        ch = info['chs'][j]
        #print(ch['ch_name'])
        #print(ch['loc'][0:3])
        ex=ch['loc'][3:6]
        ey=ch['loc'][6:9]
        ez=ch['loc'][9:12]
        R=np.vstack((ex, ey, ez))
        #take the loc points and offset by 5 mm to account for distance from scalp to sensor
        move = np.dot((0,0,offset),R)
        digpts=np.append(digpts,(ch['loc'][0:3]-move)) # to account for the gap between sensor surface and cell centre
        head_coord = mne.transforms.apply_trans(trans, ch['loc'][0:3]-move)
        digpts_head=np.append(digpts_head,(head_coord)) # to account for the gap between sensor surface and cell centre
        #digpts=np.append(digpts,ch['loc'][0:3])
        #print((ch['loc'][0:3]),(ch['loc'][0:3]-move))
        #print(move)

    n=int(digpts.shape[0]/3)
    digpts=digpts.reshape((n,3))

    n=int(digpts_head.shape[0]/3)
    digpts_head=digpts_head.reshape((n,3))

    print(digpts.shape)

    

    thesize=7

    if showDevice:
        for i in range(len(digpts)):
            ax.scatter(digpts[i][0], digpts[i][1], digpts[i][2], '10', c='cyan', alpha=0.75)
        i=0
        ax.scatter(digpts[i][0], digpts[i][1], digpts[i][2], '10', c='cyan',s =40, alpha=0.75)
        
        ax.text(digpts[i][0], digpts[i][1], digpts[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='cyan') 

    if 1:
        for i in range(len(digpts_head)):
            ax.scatter(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2], '10', c='magenta', alpha=0.75)
            if showLabels:
                ax.text(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='magenta') 

        i=0
    
        ax.scatter(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2], '10', c='magenta',s=40, alpha=0.75)
        ax.text(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='magenta') 

    LPA=raw.info['dig'][4]['r']
    NA=raw.info['dig'][3]['r']
    RPA=raw.info['dig'][5]['r']
    IN=raw.info['dig'][6]['r']
    CZ=raw.info['dig'][7]['r']

    ax.scatter(NA[0],NA[1],NA[2], '10', s=80, c='blue', marker='^', alpha=0.75)
    ax.text(NA[0],NA[1],NA[2],  'NA', size=thesize, zorder=1,  color='blue') 

    ax.scatter(LPA[0],LPA[1],LPA[2], '10', s=80, c='green', marker='^',alpha=0.75)
    ax.text(LPA[0],LPA[1],LPA[2],   'LPA', size=thesize, zorder=1,  color='green') 

    ax.scatter(RPA[0],RPA[1],RPA[2], '10', s=80, c='red', marker='^',alpha=0.75)
    ax.text(RPA[0],RPA[1],RPA[2],   'RPA', size=thesize, zorder=1,  color='red') 

    ax.scatter(IN[0],IN[1],IN[2], '10', s=80, c='black', marker='^',alpha=0.75)
    ax.scatter(CZ[0],CZ[1],CZ[2], '10', s=80, c='black', marker='^',alpha=0.75)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


# --- Main (example usage) ---------------------------------------------------
if __name__ == '__main__':
    subjects_dir = '/Users/alexandria/Documents/STANFORD/FieldLine_tests/subjects/sub-XM'
    raw_files = ['20260206_143328_sub-XM_file-xantone_raw.fif']
    ## Define constants
    trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
    
    for file in raw_files:
        if file.endswith(".fif"):
            """Do OPM-MEG load and preprocess """
            raw = mne.io.read_raw_fif(os.path.join(subjects_dir,file),'default', preload=True)
            ## Define filter type
            prepros_type = 'high-filter' 
            [raw_pre, events] = pros_OPM_data(raw, trigger_chan, prepros_type)
            
            ## Get epochs and evoked response
            tmin = -0.2  # start of each epoch (200ms before the trigger)
            tmax = 0.6  # end of each epoch (600ms after the trigger)
            epochs = mne.Epochs(raw_pre, events, tmin=tmin, tmax=tmax, preload=True)
            evoked = epochs.average()
            fig = evoked.plot_joint()
            
        elif file.endswith(".ds"):
            """ TODO: add specific CTF preprocessing after we figure out event ID issues
            Do CTF-MEG load and preprocess """
            raw = read_raw_ctf(os.path.join(subjects_dir,file), preload=True)
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
    
        
    #### STEPS TO ADD
    ## save events
    # mne.write_events( participant + '/' + participant + '_events.fif',events)
    ## define triggers
    # event_id = dict(<cond1> = 1, <cond2> = 2, <cond3> = 16, <cond4> = 32)  
    ## compute covariance and write to file
    # cov = mne.cov.compute_covariance(epochs, 0)
    # cov = mne.cov.regularize(cov, evoked.info, mag=0.05, grad = 0.05, proj = True, exclude = 'bads')
    ## make foward solution
    # fwd = mne.make_forward_solution(info = info, mri = mri, src = src, bem = bem, fname = fname, meg = True, eeg = False, overwrite = True)
    ## make inverse operator 
    # inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose = None, depth = None, fixed = False)
    ## apply inverse for each condition 
    
    
    
    
    
    
