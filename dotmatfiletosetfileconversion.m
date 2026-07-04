% SAM40 .mat -> .set conversion WITH channel locations applied
% Loads Clean_data matrices, imports into EEGLAB, attaches channel
% labels/coordinates from Coordinates.locs, and saves as .set

% Start EEGLAB without launching the GUI window
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% ---- UPDATE THESE TWO PATHS ----
data_folder = 'C:\Users\AP HAMESH\OneDrive\Desktop\AP\MIND\sam40\Data\SAM40\filtered_data';
locs_file   = 'C:\Users\AP HAMESH\OneDrive\Desktop\AP\MIND\sam40\Data\SAM40\Coordinates.locs';  % <-- point this to wherever you saved Coordinates.locs
% ---------------------------------

% Where converted .set files will be saved (kept separate from raw .mat files)
out_folder = fullfile(data_folder, 'set_converted');
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
end

% Find all task files (.mat)
files = dir(fullfile(data_folder, '*.mat'));

srate = 128; % SAM40 dataset sampling frequency (128 Hz)

fprintf('Found %d .mat files to convert.\n', length(files));

for i = 1:length(files)
    filename = files(i).name;
    [~, name_only, ~] = fileparts(filename);

    fprintf('Converting (%d/%d): %s\n', i, length(files), filename);

    % Load the file structure
    file_contents = load(fullfile(data_folder, filename));

    % Extract the 32x3200 Clean_data matrix directly
    raw_data = file_contents.Clean_data;

    % Import data array into EEGLAB format
    EEG = pop_importdata('dataformat', 'array', 'nbchan', size(raw_data,1), ...
        'data', raw_data, 'srate', srate, 'setname', name_only);

    % Attach channel locations from the .locs file (no dipfit needed)
    EEG = pop_chanedit(EEG, 'load', {locs_file, 'filetype', 'loc'});

    % Sanity check: confirm labels/positions actually got attached
    if ~isstruct(EEG.chanlocs) || isempty(EEG.chanlocs) || ~isfield(EEG.chanlocs, 'X') || isempty(EEG.chanlocs(1).X)
        warning('Channel locations did NOT attach correctly for %s. Check locs_file path.', filename);
    end

    EEG = eeg_checkset(EEG);

    % Save as a native EEGLAB file (.set) in the output folder
    EEG = pop_saveset(EEG, 'filename', [name_only '.set'], 'filepath', out_folder);
end

fprintf('Done! All files converted into EEGLAB format with channel locations.\n');
fprintf('Converted files saved in: %s\n', out_folder);
