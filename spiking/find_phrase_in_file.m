% Define the directory where your .m files are located
folder = '/home/nuttidalab/Documents/spikeRNN/analysis_code/model_eval/';  % Adjust this to your folder path

% Get a list of all .m files in the directory
files = dir(fullfile(folder, '*.m'));

% Define the phrase you want to search for
search_phrase = 'fs_ds';

% Loop through each file and search for the phrase
for k = 1:length(files)
    % Open the file for reading
    fid = fopen(fullfile(folder, files(k).name), 'rt');
    
    % Read the content of the file line by line
    line_number = 0;
    while ~feof(fid)
        line_number = line_number + 1;
        line = fgetl(fid);
        
        % Search for the phrase in the current line
        if contains(line, search_phrase)
            fprintf('Found "%s" in %s on line %d\n', search_phrase, files(k).name, line_number);
        end
    end
    
    % Close the file after reading
    fclose(fid);
end
