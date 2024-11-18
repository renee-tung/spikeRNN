

clc; clear;

normalize_ipscs = 1;
lesion_connections = 0; % if not lesioning put 0, else 'ii' etc
longer_delay = 200; % '' if standard delay (150), else a number (eg 200, 250)

models_types = {'good_models'};

if normalize_ipscs
    norm_name = '';
else
    norm_name = '_raw';
end
if ischar(lesion_connections)
    lesion_name = ['_lesion',lesion_connections];
else
    lesion_name = '';
end

for n_type = 1:length(models_types)
    models_type = models_types{n_type}; % CHANGE WHICH MODEL HERE

    clearvars -except models_types n_type models_type normalize_ipscs lesion_connections norm_name lesion_name longer_delay;

    current_path = pwd;

    task_dir = '/home/nuttidalab/Documents/spikeRNN/models/DMS_OSF';

    % Load previously saved models in the group of interest
    stable_mods = load([task_dir,'/',models_type, '/', models_type,'_list']).stable_mods;

    for n_model = 1:length(stable_mods)
        fprintf('\n');
        disp([models_types{n_type}, ' number ',num2str(n_model), ' of ', num2str(length(stable_mods))])

        % get the model
        model_name = stable_mods{n_model};

        % make a folder for the model
        cd(task_dir)
        if ~exist(model_name(1:(end-4)), 'dir')
            mkdir(model_name(1:(end-4)))
        end
        model_dir_path = strcat(task_dir,'/',model_name(1:(end-4)));
        cd(model_dir_path)

        % save name for the output
        old_name = [model_dir_path,'/','IPSCs_50travg',norm_name,lesion_name,longer_delay,'.mat'];
        new_name = [model_dir_path,'/','IPSCs_50travg',norm_name,lesion_name,num2str(longer_delay),'.mat'];

        movefile(old_name, new_name)

    end
end