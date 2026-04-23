# Copy this file to config.py and fill in your own paths.
# config.py is gitignored; each user will maintain their own local copy.

# --- Data paths --------------------------------------------------------------
sample_dir = '/path/to/your/eeg/data/'
raw_files  = ['your_raw_file.fif']
epoch_files = ['your_epoch_file.fif']
trans      = '/path/to/your/trans.fif'
dig_file   = None  # optional, set to path string if needed

# --- FreeSurfer --------------------------------------------------------------
subjects_dir = '/path/to/freesurfer/subjects/'
subject      = 'S001'

# --- Visualization & Output --------------------------------------------------------------
viz_bool = True
save_report = True
save_dir = 'path/to/your/saving/dir'
