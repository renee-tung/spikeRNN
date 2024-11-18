%% Description: script to get IPSCs avg'd over 50 trials of +1 first or -1 first

lesion_connections_list = ['ii';'ie';'ei';'ee'];

for n_connection = 1:length(lesion_connections_list)
    lesion_connections = lesion_connections_list(n_connection,:);
    disp(['lesioning ', lesion_connections])

    normalize_ipscs = 1;
    longer_delay = 400; % '' if standard delay (150), else a number (eg 200, 250)
    
    models_types = {'good_models';'bad_models'};
    % models_types = {'bad_models'};
    
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
    
        clearvars -except models_types n_type models_type normalize_ipscs lesion_connections norm_name lesion_name longer_delay lesion_connections_list;
        
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
            cd(current_path)
    
            % save name for the output
            save_name = [model_dir_path,'/','IPSCs_50travg',norm_name,lesion_name,num2str(longer_delay),'.mat'];
    
            % check if there are IPSCs calculated already
            if exist(save_name, 'file') > 0
                disp('already calculated, moving to next model...')
                continue
            end
    
            file_path = fullfile(task_dir, model_name);
            load(file_path);
        
            % dif params for model function
            use_initial_weights = false;
            scaling_factor = opt_scaling_factor;
            down_sample = 1;
            stims = struct(); 
            stims.mode = 'none'; % For LIF simulation, no stims
        
            % trial params
            % T = 411;
            stim_on = 31;
            stim_dur = 50;
            if isempty(longer_delay)
                delay = 150; % standard delay we're using
            else
                delay = longer_delay; 
            end
            T = 261+delay;
            
            fs_rate = 200;
            fs_spk = 20000;
            
            stim1_onset = (stim_on)/fs_rate*fs_spk;
            stim1_offset = (stim_on + stim_dur)/fs_rate*fs_spk;
            stim2_onset = (stim_on + stim_dur + delay)/fs_rate*fs_spk;
            stim2_offset = (stim_on + 2*stim_dur + delay)/fs_rate*fs_spk;
            baseline_onset = round(stim1_onset/2);
            
            n_trials = 50;
            ds = 1;
            
            disp('Generating IPSCs from trials...')
            % +1/+1
            ipscs_samepos = zeros(N,T*100);
            u = zeros(2, T+1); % input stim
            u(1, stim_on:stim_on+stim_dur) = 1; % first stim is +1
            u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = 1; % second stim is +1
            for i=1:n_trials
                if ischar(lesion_connections)
                    if i==1
                        disp(['lesioning ',lesion_connections, ' connections...'])
                    end
                    [~, ~, ~, ~, ~, ~, params] = LIF_network_lesion_fnc(file_path, scaling_factor,...
                        u, stims, ds, lesion_connections);
                else
                    if i==1
                        disp('not lesioning')
                    end
                    [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                        u, stims, ds, use_initial_weights);
                end
                ipscs_temp = params.IPSCs;
                if normalize_ipscs
                    if i==1
                        disp(['normalizing IPSCs...'])
                    end
                    ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2); 
                    ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                    ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
                end
                ipscs_samepos = ipscs_samepos + ipscs_temp;
                clear params ipscs_temp
            end
            ipscs_samepos = ipscs_samepos / n_trials;
    
            % -1/-1
            ipscs_sameneg = zeros(N,T*100);
            u = zeros(2, T+1); % input stim
            u(1, stim_on:stim_on+stim_dur) = -1; % first stim is -1
            u(2, stim_on+stim_dur+delay:stim_on+2*stim_dur+delay) = -1; % second stim is -1
            for i=1:n_trials
                [~, ~, ~, ~, ~, ~, params] = LIF_network_fnc(file_path, scaling_factor,...
                    u, stims, ds, use_initial_weights);
                ipscs_temp = params.IPSCs;
                if normalize_ipscs
                    ipscs_mean = mean(ipscs_temp(:,baseline_onset:stim1_onset),2);
                    ipscs_std = std(ipscs_temp(:,baseline_onset:stim1_onset),0, 2);
                    ipscs_temp = (ipscs_temp-ipscs_mean)./ipscs_std; % zscore by baseline period
                end
                ipscs_sameneg = ipscs_sameneg + ipscs_temp;
                clear params ipscs_temp
            end
            ipscs_sameneg = ipscs_sameneg / n_trials;
            
            
            save(save_name, 'ipscs_samepos', 'ipscs_sameneg', ...
                'T', 'stim_on', 'stim_dur', 'delay', 'stim1_onset', 'stim2_onset')
            
        end
    end
end