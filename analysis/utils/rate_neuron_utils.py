'''
UTIL FUNCTIONS FOR SINGLE NEURON ANALYSIS
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans
import pdb

import load_data as ld
from bootstrap_method import *




''' 
NEURON TUNING CALCULATIONS
'''


def calc_load_tuning(model_name, condn_phrase, condn_num, rates_data=None, other_variables=None, other_labels = "balance_match",
                      all_models_dir='/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5'):
    """
    for this model, get the load tuning preference for all neurons
    """
    
    if rates_data is None:
        _, rates_data = ld.load_neural_rate_data(model_name, condn_phrase, condn_num,
                                            other_labels=other_labels,
                                            all_models_dir = all_models_dir, load_LFP=False)
        # trials x neurons x time

    exc_ind, inh_ind = ld.get_celltype_label(model_name, all_models_dir=all_models_dir)

    # behavioral data
    if other_variables is not None:
        trial_labels = other_variables.get('trial_labels', None)
        trial_perfs = other_variables.get('trial_perfs', None)
        trial_outputs = other_variables.get('trial_outputs', None)
        if trial_labels is not None:
            print(f"Loaded behavioral variables from other_variables")
    else:
        trial_labels, trial_perfs, trial_outputs = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num,
                                                    other_labels=other_labels,
                                                    all_models_dir=all_models_dir)

    # get timing info
    settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num, 
                                          other_labels=other_labels,
                                          all_models_dir=all_models_dir)
    
    # each trial type
    trial_types, trial_idxs = np.unique(trial_labels[:,:2], axis=0, return_inverse=True)
    loads = np.unique(trial_types[:,0])

    # mean across time during maintenance period
    delay_means = []
    for load in loads:
        load = int(load)
        delay_idxs = [settings['stim_on'] + settings['stim_dur']*load, 
                                settings['stim_on'] + settings['stim_dur']*load + settings['delay']]
        # get mean fr during delay period for each trial
        load_idxs = np.where(trial_labels[:,0] == load)[0]
        delay_means.append(np.mean(rates_data[load_idxs, :, delay_idxs[0]:delay_idxs[1]], axis=2))
    # delay_means = np.array(delay_means) # shape: [n_loads, n_trials, n_neurons]
    
    # stats test for each neuron if there is a significant difference in firing rate 
    # between load 1 and load 3 during delay period
    tuning = np.zeros(rates_data.shape[1]) # tuning for each neuron
    for n_neuron in range(rates_data.shape[1]):
        _, p = stats.mannwhitneyu(delay_means[0][:, n_neuron], delay_means[1][:, n_neuron])
        if p < 0.01:
            tuning[n_neuron] = loads[np.argmax([delay_means[0][:, n_neuron].mean(), delay_means[1][:, n_neuron].mean()])]
        else:
            tuning[n_neuron] = np.nan
            
    return tuning


def plot_load_tuning(tuning, tuning_options = [1, 3, np.nan], cell_idxs=None, exc_ind = None, ax=None, title=None):
    """
    Plot the stim1 tuning for a given model and condition
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if cell_idxs is None:
        cell_idxs = np.arange(len(tuning))

    x_labels = [tuning_options[0], tuning_options[1],'none']
    x = np.arange(len(x_labels))
    n_tuned = np.zeros(len(x_labels))
    n_exc = np.zeros(len(x_labels))
    for i, tuning_option in enumerate(tuning_options):
        n_tuned[i] = np.sum(tuning[cell_idxs] == tuning_option)
        if np.isnan(tuning_option):
            n_tuned[i] = np.sum(np.isnan(tuning[cell_idxs]))
        if exc_ind is not None:
            n_exc[i] = np.sum(tuning[cell_idxs[exc_ind]] == tuning_option)
            if np.isnan(tuning_option):
                n_exc[i] = np.sum(np.isnan(tuning[cell_idxs[exc_ind]]))
    
    if exc_ind is None:
        ax.bar(x, n_tuned, color='black', alpha=0.5)
    else:
        ax.bar(x, n_exc, color='red', alpha=0.5)
        ax.bar(x, n_tuned-n_exc, bottom=n_exc, color='blue', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Number of neurons')
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Tuning')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Tuning of neurons to load')


def calc_stim1_tuning(model_name, condn_phrase, condn_num, other_labels = "balance_match",
                     rates_data=None, other_variables=None,
                     anova_resamples=2000, pairwise_resamples=2000,
                     use_prescreen=True, prescreen_alpha=0.20,
                     show_progress=False,
                      all_models_dir='/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5'):
    """
    for this model, get the load tuning preference for all neurons
    """
    
    if rates_data is None:
        _, rates_data = ld.load_neural_rate_data(model_name, condn_phrase, condn_num,
                                            other_labels=other_labels,
                                            all_models_dir = all_models_dir, load_LFP=False)
        # trials x neurons x time

    # Unpack behavioral variables from other_variables if provided
    if other_variables is not None:
        trial_labels = other_variables.get('trial_labels', None)
        trial_perfs = other_variables.get('trial_perfs', None)
        trial_outputs = other_variables.get('trial_outputs', None)
        if trial_labels is not None:
            print(f"Loaded behavioral variables from other_variables")
    else:
        trial_labels = trial_perfs = trial_outputs = None

    # Load from file if not provided
    if trial_labels is None:
        print('Loading behavioral data...')
        trial_labels, trial_perfs, trial_outputs = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num,
                                                        other_labels=other_labels,
                                                        all_models_dir=all_models_dir)

    # get timing info
    settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num, 
                                          other_labels=other_labels, all_models_dir=all_models_dir)

    # Get stim1 label column. For sternberg trial labels this is expected to be col 2.
    stim1_col = 2 
    stim1_labels = trial_labels[:, stim1_col]
    stim1s = np.unique(stim1_labels)

    # Mean firing rate during stim1 period for each trial and neuron: [n_trials, n_neurons]
    stim1_idx_start = int(settings['stim_on'])
    stim1_idx_stop = int(settings['stim_on'] + settings['stim_dur'])
    stim1_trial_means = np.mean(rates_data[:, :, stim1_idx_start:stim1_idx_stop], axis=2)

    # Group trial-wise means by stim1 identity.
    # Each entry has shape [n_trials_for_stim, n_neurons].
    stim1_groups = [stim1_trial_means[stim1_labels == stim1] for stim1 in stim1s]

    if len(stim1_groups) < 2:
        return np.full(rates_data.shape[1], np.nan)

    # Parametric pre-screen + permutation one-way ANOVA + max-vs-others permutation t-tests.
    tuning = np.full(rates_data.shape[1], np.nan)

    def f_statistic(*samples):
        return stats.f_oneway(*samples).statistic

    def t_statistic(x, y):
        return stats.ttest_ind(x, y, equal_var=False).statistic

    def pvalue_greater_from_ttest(x, y):
        t_stat, p_two_sided = stats.ttest_ind(x, y, equal_var=False)
        if np.isnan(t_stat) or np.isnan(p_two_sided):
            return np.nan
        if t_stat > 0:
            return p_two_sided / 2.0
        return 1.0 - (p_two_sided / 2.0)

    neuron_iter = range(rates_data.shape[1])
    if show_progress:
        try:
            tqdm_module = __import__('tqdm.auto', fromlist=['tqdm'])
            neuron_iter = tqdm_module.tqdm(neuron_iter, desc='Stim1 tuning', leave=False)
        except Exception:
            pass

    for n_neuron in neuron_iter:
        neuron_groups = [group[:, n_neuron] for group in stim1_groups]

        # Skip if any group is empty or contains only NaNs.
        if any(g.size == 0 or np.all(np.isnan(g)) for g in neuron_groups):
            continue

        neuron_groups = [g[~np.isnan(g)] for g in neuron_groups]
        if any(g.size == 0 for g in neuron_groups):
            continue

        group_means = np.array([g.mean() for g in neuron_groups])
        max_idx = int(np.argmax(group_means))
        max_group = neuron_groups[max_idx]

        if use_prescreen:
            try:
                param_anova_p = stats.f_oneway(*neuron_groups).pvalue
            except Exception:
                continue

            if np.isnan(param_anova_p) or param_anova_p >= prescreen_alpha:
                continue

            prescreen_pairwise_ok = True
            for i_group, other_group in enumerate(neuron_groups):
                if i_group == max_idx:
                    continue
                p_greater = pvalue_greater_from_ttest(max_group, other_group)
                if np.isnan(p_greater) or p_greater >= prescreen_alpha:
                    prescreen_pairwise_ok = False
                    break

            if not prescreen_pairwise_ok:
                continue

        anova_perm = stats.permutation_test(
            tuple(neuron_groups),
            f_statistic,
            permutation_type='independent',
            alternative='greater',
            n_resamples=anova_resamples,
        )

        if anova_perm.pvalue >= 0.05:
            continue

        all_pairwise_sig = True
        for i_group, other_group in enumerate(neuron_groups):
            if i_group == max_idx:
                continue

            # Original mean-difference permutation test (kept for reference):
            # pairwise_perm = stats.permutation_test(
            #     (max_group, other_group),
            #     lambda x, y: np.mean(x) - np.mean(y),
            #     permutation_type='independent',
            #     alternative='greater',
            #     n_resamples=5000,
            # )

            pairwise_perm = stats.permutation_test(
                (max_group, other_group),
                t_statistic,
                permutation_type='independent',
                alternative='greater',
                n_resamples=pairwise_resamples,
            )

            if pairwise_perm.pvalue >= 0.05:
                all_pairwise_sig = False
                break

        if all_pairwise_sig:
            tuning[n_neuron] = stim1s[max_idx]
            
    return tuning


def _holm_adjust_pvalues(pvals):
    """Holm-Bonferroni adjustment for a 1D array of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return pvals

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    adjusted_sorted = np.empty(m, dtype=float)
    for i in range(m):
        adjusted_sorted[i] = (m - i) * sorted_p[i]
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _dunn_test_pval_matrix(groups, p_adjust='holm'):
    """
    Dunn's post-hoc test for multiple independent groups.

    Parameters
    ----------
    groups : list of 1D ndarray
        Grouped samples (NaNs should already be removed).
    p_adjust : str
        Multiple-comparison correction method; currently supports 'holm' or None.

    Returns
    -------
    pval_mat : ndarray, shape [k, k]
        Symmetric pairwise p-value matrix with NaN on diagonal.
    """
    k = len(groups)
    pval_mat = np.full((k, k), np.nan, dtype=float)
    if k < 2:
        return pval_mat

    sizes = np.array([g.size for g in groups], dtype=float)
    if np.any(sizes == 0):
        return pval_mat

    pooled = np.concatenate(groups)
    n_total = pooled.size
    if n_total < 2:
        return pval_mat

    ranks = stats.rankdata(pooled, method='average')

    # Tie correction for Dunn variance term.
    _, tie_counts = np.unique(pooled, return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts)
    denom = (n_total**3 - n_total)
    tie_correction = 1.0 if denom == 0 else 1.0 - (tie_term / denom)
    if tie_correction <= 0:
        return pval_mat

    rank_means = np.zeros(k, dtype=float)
    start = 0
    for i, size in enumerate(sizes.astype(int)):
        stop = start + size
        rank_means[i] = np.mean(ranks[start:stop])
        start = stop

    pair_indices = []
    pair_pvals = []
    base = (n_total * (n_total + 1.0) / 12.0) * tie_correction

    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(base * (1.0 / sizes[i] + 1.0 / sizes[j]))
            if se == 0:
                p_ij = np.nan
            else:
                z = np.abs(rank_means[i] - rank_means[j]) / se
                p_ij = 2.0 * (1.0 - stats.norm.cdf(z))
            pair_indices.append((i, j))
            pair_pvals.append(p_ij)

    pair_pvals = np.asarray(pair_pvals, dtype=float)
    valid = ~np.isnan(pair_pvals)

    if p_adjust == 'holm':
        adjusted = pair_pvals.copy()
        adjusted[valid] = _holm_adjust_pvalues(pair_pvals[valid])
    else:
        adjusted = pair_pvals

    for (i, j), p_ij in zip(pair_indices, adjusted):
        pval_mat[i, j] = p_ij
        pval_mat[j, i] = p_ij

    return pval_mat


def calc_stim1_tuning_nonparametric(model_name, condn_phrase, condn_num, other_labels="balance_match",
                                    rates_data=None, other_variables=None,
                                    use_prescreen=True, prescreen_alpha=0.20,
                                    kw_alpha=0.05, dunn_alpha=0.05,
                                    dunn_p_adjust='holm',
                                    show_progress=False,
                                    all_models_dir='/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5'):
    """
    Nonparametric stim1 tuning preference for all neurons.

    Uses Kruskal-Wallis as omnibus test, then Dunn's post-hoc pairwise tests
    (with Holm correction by default). A neuron is tuned if the group with the
    highest mean firing rate differs significantly from all other groups.
    """

    if rates_data is None:
        _, rates_data = ld.load_neural_rate_data(model_name, condn_phrase, condn_num,
                                                 other_labels=other_labels,
                                                 all_models_dir=all_models_dir, load_LFP=False)
        # trials x neurons x time

    # Unpack behavioral variables from other_variables if provided.
    if other_variables is not None:
        trial_labels = other_variables.get('trial_labels', None)
        trial_perfs = other_variables.get('trial_perfs', None)
        trial_outputs = other_variables.get('trial_outputs', None)
        if trial_labels is not None:
            print("Loaded behavioral variables from other_variables")
    else:
        trial_labels = trial_perfs = trial_outputs = None

    # Load from file if not provided.
    if trial_labels is None:
        print('Loading behavioral data...')
        trial_labels, trial_perfs, trial_outputs = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num,
                                                                          other_labels=other_labels,
                                                                          all_models_dir=all_models_dir)

    # Get timing info.
    settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num,
                                          other_labels=other_labels,
                                          all_models_dir=all_models_dir)

    # Get stim1 label column. For sternberg trial labels this is expected to be col 2.
    stim1_col = 2
    stim1_labels = trial_labels[:, stim1_col]
    stim1s = np.unique(stim1_labels)

    # Mean firing rate during stim1 period for each trial and neuron: [n_trials, n_neurons].
    stim1_idx_start = int(settings['stim_on'])
    stim1_idx_stop = int(settings['stim_on'] + settings['stim_dur'])
    stim1_trial_means = np.mean(rates_data[:, :, stim1_idx_start:stim1_idx_stop], axis=2)

    # Group trial-wise means by stim1 identity.
    stim1_groups = [stim1_trial_means[stim1_labels == stim1] for stim1 in stim1s]

    if len(stim1_groups) < 2:
        return np.full(rates_data.shape[1], np.nan)

    tuning = np.full(rates_data.shape[1], np.nan)

    neuron_iter = range(rates_data.shape[1])
    if show_progress:
        try:
            tqdm_module = __import__('tqdm.auto', fromlist=['tqdm'])
            neuron_iter = tqdm_module.tqdm(neuron_iter, desc='Stim1 tuning (nonparametric)', leave=False)
        except Exception:
            pass

    for n_neuron in neuron_iter:
        neuron_groups = [group[:, n_neuron] for group in stim1_groups]

        # Skip if any group is empty or contains only NaNs.
        if any(g.size == 0 or np.all(np.isnan(g)) for g in neuron_groups):
            continue

        neuron_groups = [g[~np.isnan(g)] for g in neuron_groups]
        if any(g.size == 0 for g in neuron_groups):
            continue

        group_means = np.array([g.mean() for g in neuron_groups])
        max_idx = int(np.argmax(group_means))

        try:
            kw_p = stats.kruskal(*neuron_groups).pvalue
        except Exception:
            continue

        if np.isnan(kw_p):
            continue

        if use_prescreen and kw_p >= prescreen_alpha:
            continue

        if kw_p >= kw_alpha:
            continue

        dunn_pvals = _dunn_test_pval_matrix(neuron_groups, p_adjust=dunn_p_adjust)

        max_vs_others = np.delete(dunn_pvals[max_idx, :], max_idx)
        if max_vs_others.size == 0 or np.any(np.isnan(max_vs_others)):
            continue

        if np.all(max_vs_others < dunn_alpha):
            tuning[n_neuron] = stim1s[max_idx]

    return tuning


def plot_stim1_tuning(tuning, tuning_options = [0, 1, 2, 3, np.nan], cell_idxs=None, exc_ind = None, ax=None, title=None):
    """
    Plot the stim1 tuning for a given model and condition
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if cell_idxs is None:
        cell_idxs = np.arange(len(tuning))

    x_labels = [tuning_options[0], tuning_options[1], tuning_options[2], tuning_options[3], 'none']
    x = np.arange(len(x_labels))
    n_tuned = np.zeros(len(x_labels))
    n_exc = np.zeros(len(x_labels))
    for i, tuning_option in enumerate(tuning_options):
        n_tuned[i] = np.sum(tuning[cell_idxs] == tuning_option)
        if np.isnan(tuning_option):
            n_tuned[i] = np.sum(np.isnan(tuning[cell_idxs]))
        if exc_ind is not None:
            n_exc[i] = np.sum(tuning[cell_idxs[exc_ind]] == tuning_option)
            if np.isnan(tuning_option):
                n_exc[i] = np.sum(np.isnan(tuning[cell_idxs[exc_ind]]))
    
    if exc_ind is None:
        ax.bar(x, n_tuned, color='black', alpha=0.5)
    else:
        ax.bar(x, n_exc, color='red', alpha=0.5)
        ax.bar(x, n_tuned-n_exc, bottom=n_exc, color='blue', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Number of neurons')
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Tuning')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Tuning of neurons to stimulus 1 identity')



'''
FIRING RATE FUNCTIONS
'''

def plot_neuron_rates(model_name, cell_id, condn_phrase, condn_num, rates_data = None, other_labels=[], cut_off = 50, 
                      ax=None, title=None, all_models_dir='/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5'):
    """
    Plot the firing rates of a neuron across trials.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    - rates_data: shape [trials, neurons, time]
    """
    # Load the firing rates
    if rates_data is None:
        _, rates_data = ld.load_neural_rate_data(model_name, condn_phrase, condn_num, other_labels=other_labels,
                                            all_models_dir = all_models_dir, load_LFP=False)

    exc_ind, inh_ind = ld.get_celltype_label(model_name, all_models_dir=all_models_dir)
    cell_type = 'exc' if cell_id in exc_ind else 'inh'

    # behavioral data
    trial_labels, trial_perfs, trial_outputs = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num, other_labels=other_labels,
                                                    all_models_dir=all_models_dir)

    # get timing info
    settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num, all_models_dir=all_models_dir)

    _, colors = get_trialtype_colors()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels[:,:2], axis=0, return_inverse=True)
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[trials_idx, cell_id, cut_off:].T  # shape: [time, trial]
        # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

        # first just plot mean and sem
        mean_rate = np.mean(trials_rate, axis=1)
        sem_rate = stats.sem(trials_rate, axis=1)

        ax.plot(np.arange(cut_off,settings['T']), mean_rate, color=colors[i], label=f'{trial_type}')
        ax.fill_between(np.arange(cut_off,settings['T']), mean_rate-sem_rate, mean_rate+sem_rate, color=colors[i], alpha=0.2)

    # shade stim times
    loads = np.unique(trial_types[:,0])
    # colors = ['gray','red']
    for i_load, load in enumerate(loads):
        ax.axvspan(settings['stim_on'], settings['stim_on'] + settings['stim_dur']*load, color=colors[i_load*2], alpha=0.2)
        ax.axvspan(settings['stim_on'] + settings['stim_dur']*load + settings['delay'], settings['stim_on'] + settings['stim_dur']*load + settings['delay'] + settings['stim_dur'], color=colors[i_load*2], alpha=0.2)

    # axes
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    if title is not None:
        ax.set_title(f'{title}')
    else:
        ax.set_title(f'Avg cell {cell_id} fr ({cell_type})')
    ax.legend()
    ax.set_xlim(0, settings['T'])


def plot_rates(rates_data, idxs_one, idxs_two, settings, trial_labels,
               ax = None, title=None, labels_one='Group 1', labels_two='Group 2', 
               colors=None, ):
    
    # rates_data shape: [trials/neurons, time]
    # idxs_one and idxs_two are indices indicating which trials/neurons belong to each group
    
    g1_mean = np.nanmean(rates_data[idxs_one,:], axis=0)
    g1_sem = stats.sem(rates_data[idxs_one,:], axis=0)
    g2_mean = np.nanmean(rates_data[idxs_two,:], axis=0)
    g2_sem = stats.sem(rates_data[idxs_two,:], axis=0)
    
    if colors is None:
        _, colors = get_trialtype_colors()
        colors = [colors[0], colors[2]]
    
    t_range = np.arange(rates_data.shape[1])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(g1_mean, color=colors[0], label=[labels_one + f' (n={len(idxs_one)})'])
    ax.fill_between(t_range, g1_mean-g1_sem, g1_mean+g1_sem, color=colors[0], alpha=0.2)
    ax.plot(g2_mean, color=colors[1], label=[labels_two + f' (n={len(idxs_two)})'])
    ax.fill_between(t_range, g2_mean-g2_sem, g2_mean+g2_sem, color=colors[1], alpha=0.2)
    
    # shade stim times
    loads = np.unique(trial_labels[:,0])
    for i_load, load in enumerate(loads):
        ax.axvspan(settings['stim_on'], settings['stim_on'] + settings['stim_dur']*load, color=colors[i_load], alpha=0.2)
        ax.axvspan(settings['stim_on'] + settings['stim_dur']*load + settings['delay'], settings['stim_on'] + settings['stim_dur']*load + settings['delay'] + settings['stim_dur'], color=colors[i_load], alpha=0.2)
        
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    if title is not None:
        ax.set_title(f'{title}')
    ax.legend()
    ax.set_xlim(0, settings['T'])
    
    return ax


# def plot_neuron_rates_by_acc(model_name, cell_idx, condn_phrase, condn_num, rates_data = None, cut_off = 50, ax=None, title=None):
#     """
#     Plot the firing rates of a neuron across trials.
    
#     Parameters:
#     - model_name: Name of the model.
#     - cell_id: ID of the neuron.
#     - condn_phrase: Condition phrase for loading data.
#     - condn_num: Condition number for loading data.
#     """
#     # Load the firing rates
#     if rates_data is None:
#         _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
#                                                     load_LFP=False, load_spikes=False, load_rates=True)

#     exc_ind, inh_ind = ld.get_celltype_label(model_name)
#     cell_type = 'exc' if cell_idx in exc_ind else 'inh'

#     # behavioral data
#     trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

#     # get timing info
#     times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms
#     settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num, all_models_dir=all_models_dir)

#     _, colors = get_trialtype_colors()
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 4))

#     # plot firing rate for each trial type
#     trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    
#     for i, trial_type in enumerate(trial_types):
#         trials_idx = (trial_idxs == i)
#         trials_rate = rates_data[cut_off:, cell_idx, trials_idx]
#         # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

#         # get correct and incorrect trials
#         correct_trials = np.where(trial_perfs[trials_idx] == 1)[0]
#         incorrect_trials = np.where(trial_perfs[trials_idx] == 0)[0]

#         mean_corr = np.mean(trials_rate[:, correct_trials], axis=1)
#         sem_corr = stats.sem(trials_rate[:, correct_trials], axis=1)
#         mean_inc = np.mean(trials_rate[:, incorrect_trials], axis=1)
#         sem_inc = stats.sem(trials_rate[:, incorrect_trials], axis=1)
#         # plot firing rate for correct trials
#         ax.plot(np.arange(cut_off,times_ms['T']), mean_corr, color=colors[i], label=f'{trial_type} correct')
#         ax.fill_between(np.arange(cut_off,times_ms['T']), mean_corr-sem_corr, mean_corr+sem_corr, color=colors[i], alpha=0.2)
#         # plot firing rate for incorrect trials
#         ax.plot(np.arange(cut_off,times_ms['T']), mean_inc, color=colors[i], linestyle='--', label=f'{trial_type} incorrect')
#         ax.fill_between(np.arange(cut_off,times_ms['T']), mean_inc-sem_inc, mean_inc+sem_inc, color=colors[i], alpha=0.2)

#     # shade stim times
#     ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
#     ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)

#     # axes
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Firing rate (Hz)')
#     if title is not None:
#         ax.set_title(f'{title}')
#     else:
#         ax.set_title(f'Cell {cell_idx} fr ({cell_type})')
#     # ax.legend()
#     ax.set_xlim(0, times_ms['T'])


# def plot_neuron_rates_by_acc_bootstrap(model_name, cell_idx, condn_phrase, condn_num, rates_data = None, 
#                                        nboot=1000, CI_int=(2.5, 97.5), random_seed=42,
#                                        cut_off = 50, ax=None, title=None, plot=1):
#     """
#     Plot the firing rates of a neuron across trials.
    
#     Parameters:
#     - model_name: Name of the model.
#     - cell_id: ID of the neuron.
#     - condn_phrase: Condition phrase for loading data.
#     - condn_num: Condition number for loading data.
#     """
#     # Load the firing rates
#     if rates_data is None:
#         _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
#                                                     load_LFP=False, load_spikes=False, load_rates=True)

#     exc_ind, inh_ind = ld.get_celltype_label(model_name)
#     cell_type = 'exc' if cell_idx in exc_ind else 'inh'

#     # behavioral data
#     trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

#     # get timing info
#     times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

#     _, colors = get_trialtype_colors()
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 4))

#     # plot firing rate for each trial type
#     trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
    
#     for i, trial_type in enumerate(trial_types):
#         trials_idx = (trial_idxs == i)
#         trials_rate = rates_data[cut_off:, cell_idx, trials_idx]
#         # trials_rate = rates_data[cut_off:, cell_id, trials_idx]

#         # get correct and incorrect trials
#         correct_trials = np.where(trial_perfs[trials_idx] == 1)[0]
#         incorrect_trials = np.where(trial_perfs[trials_idx] == 0)[0]
        
#         # bootstrap method
#         t1_avg, t1_CI, t2_avg, t2_CI, diff_avg, diff_CI, p_diff = fnc_time_bootstrap_optimized_retX(trials_rate[:, correct_trials].T, 
#                                                                                        trials_rate[:, incorrect_trials].T, 
#                                                                                        nboot, CI_int, random_seed=random_seed)
#         # # now plot
#         # ax.plot(np.arange(cut_off,times_ms['T']), t1_avg, color=colors[i], label=f'{trial_type} correct')
#         # ax.fill_between(np.arange(cut_off,times_ms['T']), t1_CI[:,0], t1_CI[:,1], color=colors[i], alpha=0.2)
#         # ax.plot(np.arange(cut_off,times_ms['T']), t2_avg, color=colors[i], linestyle='--', label=f'{trial_type} incorrect')
#         # ax.fill_between(np.arange(cut_off,times_ms['T']), t2_CI[:,0], t2_CI[:,1], color=colors[i], alpha=0.2)

#         # now plot difference
#         ax.plot(np.arange(cut_off,times_ms['T']), diff_avg, color=colors[i], label=f'{trial_type} diff')
#         ax.fill_between(np.arange(cut_off,times_ms['T']), diff_CI[:,0], diff_CI[:,1], color=colors[i], alpha=0.2)
#         # print p-value
#         print(f'Cell {cell_idx}, trial type {trial_type}, p-value: {p_diff.mean()}')


#     # shade stim times
#     ax.axvspan(times_ms['stim1_on'], times_ms['stim1_off'], color='gray', alpha=0.2)
#     ax.axvspan(times_ms['stim2_on'], times_ms['stim2_off'], color='gray', alpha=0.2)
#     # axes
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Firing rate (Hz)')
#     if title is not None:
#         ax.set_title(f'{title}')
#     else:
#         ax.set_title(f'Cell {cell_idx} fr ({cell_type})')
#     # ax.legend()
#     ax.set_xlim(0, times_ms['T'])
    
#     # return t1_CI, t2_CI, t1_avg, t2_avg, p_diff


def plot_trialtype_meanrates(model_name, condn_phrase, condn_num, rates_data = None, sort=None,
                     cut_off = 25, hlines = [], normalize=False, 
                     all_models_dir = '/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5', 
                     vmin=0, vmax=1, cmap='Greys'):
    """
    Plot the mean firing rates of all neurons across all trial types.
    
    Parameters:
    - model_name: Name of the model.
    - cell_id: ID of the neuron.
    - condn_phrase: Condition phrase for loading data.
    - condn_num: Condition number for loading data.
    """
    
    if rates_data is None:
        _, rates_data = ld.load_neural_rate_data(model_name, condn_phrase, condn_num,
                                            all_models_dir = all_models_dir, load_LFP=False)
        
    n_cells = rates_data.shape[1]

    exc_ind, inh_ind = ld.get_celltype_label(model_name, all_models_dir=all_models_dir)
    
    # behavioral data
    trial_labels, trial_perfs, trial_outputs = ld.load_bhv_rate_data(model_name, condn_phrase, condn_num,
                                                    all_models_dir=all_models_dir)

    # get timing info
    settings = ld.load_settings_rate_data(model_name, condn_phrase, condn_num, all_models_dir=all_models_dir)

    _, colors = get_trialtype_colors()
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(8, 4))

    # plot firing rate for each trial type
    trial_types, trial_idxs = np.unique(trial_labels[:,:2], axis=0, return_inverse=True)
    cell_meanfrs = np.zeros((n_cells, len(trial_types), int(settings['T'] - cut_off)))
    for i, trial_type in enumerate(trial_types):
        trials_idx = (trial_idxs == i)
        trials_rate = rates_data[trials_idx, :, cut_off:]  # shape: [trials, neurons, time]
        cell_meanfrs[:, i, :] = np.mean(trials_rate, axis=0)  # mean over trials, transpose to match shape

    if sort is None:
        sort = np.arange(n_cells)

    if normalize:
        print(cell_meanfrs.shape)
        cell_meanfrs = stats.zscore(cell_meanfrs, axis=2)
        vmin=-2; vmax=2; cmap='bwr'
        # baseline = np.mean(cell_meanfrs[:,:,cut_off:int(times_ms['stim1_on'])], axis=2)
        # cell_meanfrs = (cell_meanfrs - baseline[:,:,np.newaxis])
        # vmin=-5; vmax=5; cmap='bwr'
    else:
        cmap='Greys'
           
    
    fig, axs = plt.subplots(2,2, figsize=(16, 8))
    axs = axs.flatten()
    for i, trial_type in enumerate(trial_types):
        axs[i].imshow(cell_meanfrs[sort, i, :], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(f'Trial type {trial_type}')
        axs[i].set_xlabel('Time (ms)')
        axs[i].set_ylabel('Cell ID')
        axs[i].set_xticks(np.arange(cut_off, settings['T'], 200))
        axs[i].set_xticklabels(np.arange(cut_off, settings['T'], 200))
        axs[i].set_yticks(np.arange(0, n_cells, 100))
        axs[i].set_yticklabels(np.arange(0, n_cells, 100))
        axs[i].set_xlim(0, settings['T'] - cut_off)
        axs[i].set_ylim(0, n_cells)

        if len(hlines) > 0:
            for hline in hlines:
                axs[i].axhline(y=hline, color='k', linestyle='--')

        load = int(trial_type[0])

        # shade stimulus periods
        # stim_colors = get_stim_plotting_colors(trial_type)
        axs[i].axvspan(settings['stim_on']-cut_off, settings['stim_on']-cut_off + settings['stim_dur']*load, color=colors[load], alpha=0.3)
        axs[i].axvspan(settings['stim_on']-cut_off + settings['stim_dur']*load + settings['delay'], 
                       settings['stim_on']-cut_off + settings['stim_dur']*load + settings['delay'] + settings['stim_dur'], color=colors[load], alpha=0.3)


    plt.tight_layout()
    plt.show()


# def plot_trialtype_meanrates_by_acc(model_name, condn_phrase, condn_num, rates_data = None, sort=None,
#                      cut_off = 50, ax=None, title=None):
#     """
#     Plot the mean firing rates of all neurons across all trial types.
    
#     Parameters:
#     - model_name: Name of the model.
#     - cell_id: ID of the neuron.
#     - condn_phrase: Condition phrase for loading data.
#     - condn_num: Condition number for loading data.
#     """
#     # Load the firing rates
#     if rates_data is None:
#         _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
#                                                     load_LFP=False, load_spikes=False, load_rates=True)
#     n_cells = rates_data.shape[1]

#     # behavioral data
#     trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

#     # get timing info
#     times_ms, times_real, fs_dict = ld.get_times_dict('ds', condn_phrase, condn_num) #ds is fs=1000, ms

#     # plot firing rate for each trial type
#     trial_types, trial_idxs = np.unique(trial_labels, axis=0, return_inverse=True)
#     cell_meanfrs_corr = np.zeros((n_cells, len(trial_types), int(times_ms['T'] - cut_off)))
#     cell_meanfrs_inc = np.zeros((n_cells, len(trial_types), int(times_ms['T'] - cut_off)))

#     for i, trial_type in enumerate(trial_types):
#         # get correct and incorrect trials
#         trials_idx_corr = np.where((trial_idxs == i) & (trial_perfs == 1))[0]
#         trials_idx_inc = np.where((trial_idxs == i) & (trial_perfs == 0))[0]
        
#         cell_meanfrs_corr[:,i,:] = np.mean(rates_data[cut_off:, :, trials_idx_corr], axis=2).T  # mean over trials, transpose to match shape
#         cell_meanfrs_inc[:,i,:] = np.mean(rates_data[cut_off:, :, trials_idx_inc], axis=2).T  # mean over trials, transpose to match shape

#     if sort is None:
#         sort = np.arange(n_cells)
    
#     fig, axs = plt.subplots(3,4, figsize=(16, 12))
#     for i, trial_type in enumerate(trial_types):
#         axs[0,i].imshow(cell_meanfrs_corr[sort, i, :], aspect='auto', cmap='Grays', vmin=0, vmax=40)
#         axs[1,i].imshow(cell_meanfrs_inc[sort, i, :], aspect='auto', cmap='Grays', vmin=0, vmax=40)
#         axs[2,i].imshow(cell_meanfrs_corr[sort, i, :] - cell_meanfrs_inc[sort, i, :], aspect='auto', cmap='bwr', vmin=-8, vmax=8)
#         axs[0,i].set_title(f'Trial type {trial_type} correct')
#         axs[1,i].set_title(f'Trial type {trial_type} incorrect')
#         axs[2,i].set_title(f'Trial type {trial_type} diff (corr - inc)')
#         for j in range(3):
#             # shade stimulus periods
#             stim_colors = get_stim_plotting_colors(trial_type)
#             axs[j,i].axvspan(times_ms['stim1_on']-cut_off, times_ms['stim1_off']-cut_off, color=stim_colors[0], alpha=0.3)
#             axs[j,i].axvspan(times_ms['stim2_on']-cut_off, times_ms['stim2_off']-cut_off, color=stim_colors[1], alpha=0.3)
#         axs[j,i].set_xlabel('Time (ms)')
#         axs[j,i].set_ylabel('Cell ID')
#         axs[j,i].set_xticks(np.arange(cut_off, times_ms['T'], 500))
#         axs[j,i].set_xticklabels(np.arange(cut_off, times_ms['T'], 500))
#         axs[j,i].set_yticks(np.arange(0, n_cells, 10))
#         # axs[i,j].set_yticklabels(np.arange(0, n_cells, 10))
#         axs[j,i].set_xlim(0, times_ms['T'] - cut_off)
#         axs[j,i].set_ylim(0, n_cells)
#     plt.tight_layout()
#     plt.show()



'''
FUNCTIONAL SUBPOPULATION FUNCTIONS
'''

def calc_subpop(model_name, condn_phrase, condn_num, method= 'rate_dist', rates_data=None, plot=False):
    """
    for this model + condition, get the subpopulation preference for all neurons
    method = 'rate_dist' or 'tuning'

    """
    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    # behavioral data
    trial_labels, trial_perfs = ld.load_bhv_data(model_name, condn_phrase, condn_num)

    # timing data
    times_ms, _,_ = ld.get_times_dict('ds', condn_phrase, condn_num)


    if method == 'rate_dist':
        # restructure rate data 
        # (matrix of just mean fr for each neuron for each trial type: time x neurons x 4 trial types), excluding fixation period
        stim1_on = int(times_ms['stim1_on'])
        t = int(rates_data.shape[0] - stim1_on)
        baseline_idx = [int(times_ms['stim1_on']/2), int(times_ms['stim1_on'])] 
        rates_data = baseline_norm_rate(rates_data, baseline_idx=baseline_idx) # baseline-norm each trial
        rates_mean = np.zeros((t, rates_data.shape[1], 4))
        trialtype_idxs = np.zeros((t*4))
        trial_types, trial_idxs = np.unique(trial_labels[:,:2], axis=0, return_inverse=True)
        for i, trial_type in enumerate(trial_types):
            trials_idx = (trial_idxs == i)
            trialtype_idxs[i*t:(i+1)*t] = i
            # get mean rate
            rates_mean[:,:, i] = np.mean(rates_data[stim1_on:, :, trials_idx], axis=2)
        rates_mean = rates_mean.transpose(2,0,1).reshape(-1, rates_mean.shape[1]) # concatenate 4 trial types one after another in time
        N = rates_mean.shape[1]

        # z-score the rates, then get distances
        # rate_features = get_rate_features(rates_mean.T, normalize=True) #zscore
        # rate_dist = pairwise_distances(rate_features, metric='euclidean')
        rate_dist = pairwise_distances(rates_mean, metric='euclidean')
        rate_dist /= np.max(rate_dist)

        # UMAP on the rate distances
        embedding = embed_umap(rate_dist, dim=20)

        # KMeans on the embedding
        k, labels = get_kmeans_clusters(embedding, n_clusters=None, plot=plot)
        
    elif method == 'tuning':
        labels = calc_stim1_tuning(model_name, condn_phrase, condn_num, rates_data=rates_data)

    else:
        raise ValueError("Method not defined, must be 'rate_dist' or 'tuning'")

    return labels


def baseline_norm_rate(r, baseline_idx=None):
    '''
    normalize firing rates to z-scores
    r: firing rates, shape (n_neurons, T), or (T, n_neurons, n_trials)
    baseline: if None, use the mean of the first 100 ms as baseline
    '''
    n_dims = len(r.shape)
    if n_dims == 2:
        # r is (n_neurons, T)
        if baseline_idx is None:
            baseline_mean = r[:, :100].mean(axis=1, keepdims=True)
            baseline_std = r[:, :100].std(axis=1, keepdims=True)
        else:
            baseline_mean = r[:, baseline_idx[0]:baseline_idx[1]].mean(axis=1, keepdims=True)
            baseline_std = r[:, baseline_idx[0]:baseline_idx[1]].std(axis=1, keepdims=True)
        # do z-score normalization
        r = (r - baseline_mean) / (baseline_std + 1e-10)

    elif n_dims == 3:
        # r is (T, n_neurons, n_trials)
        if baseline_idx is None:
            baseline_mean = r[:100,:,:].mean(axis=0, keepdims=True)
            baseline_std = r[:100,:,:].std(axis=0, keepdims=True)
        else:
            baseline_mean = r[baseline_idx[0]:baseline_idx[1],:,:].mean(axis=0, keepdims=True)
            baseline_std = r[baseline_idx[0]:baseline_idx[1],:,:].std(axis=0, keepdims=True)
        # do z-score normalization
        r = (r - baseline_mean) / (baseline_std + 1e-10)

    return r 



def get_rate_features(r, normalize=True):
    '''
    normalize firing rates to z-scores
    r: firing rates, shape (n_neurons, T)
    normalize: if True, normalize to z-scores
    '''
    if normalize:
        r = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-8)
    return r  # shape (200, T)


def embed_umap(dist_matrix, dim=2):
    umap = UMAP(n_components=dim, random_state=42)
    embedding = umap.fit_transform(dist_matrix)
    return embedding


def get_kmeans_clusters(embedding, n_clusters=None, cluster_range=range(2,11), plot=False):
    """
    Get the KMeans clusters for the given embedding.
    
    Parameters:
    - embedding: The UMAP embedding of the data.
    - n_clusters: The number of clusters to use. If None, it will be determined using the silhouette score.
    
    Returns:
    - n_clusters: The best number of clusters determined by silhouette score (or provided).
    - labels: The cluster labels for each point in the embedding.
    """
    if n_clusters is None:
        silhouette_scores = []
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(embedding)
            silhouette_scores.append(silhouette_score(embedding, labels))
        n_clusters = np.argmax(silhouette_scores) + cluster_range.start

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(cluster_range, silhouette_scores)
        axs[0].axvline(x=n_clusters, color='r', linestyle='--')
        axs[0].set_title("Silhouette Score vs. Number of Clusters")
        axs[0].set_xlabel("Number of Clusters")
        axs[0].set_ylabel("Silhouette Score")
        axs[0].grid(True)
        
        scatter = axs[1].scatter(embedding[:, 0], embedding[:, 1], c=labels, s=30)
        axs[1].set_title(f"KMeans Clustering")
        axs[1].set_xlabel("Dim 1")
        axs[1].set_ylabel("Dim 2")
        axs[1].grid(True)
        plt.colorbar(scatter)

    return n_clusters, labels


def plot_rates_by_cluster(model_name, condn_phrase, condn_num, labels, rates_data=None):

    if rates_data is None:
        _, _, rates_data = ld.load_neural_data(model_name, condn_phrase, condn_num,
                                                    load_LFP=False, load_spikes=False, load_rates=True)
    sort = np.argsort(labels)
    _, hlines = np.unique(np.sort(labels), return_index=True)
    plot_trialtype_meanrates(model_name, condn_phrase, condn_num, rates_data = rates_data, cut_off=50, sort=sort, normalize=True,
                                hlines = hlines)


'''
UTILITY FUNCTIONS
'''

def get_stim_plotting_colors(stim):
    stim1_color = 'b' if stim[0] == 1 else 'r'
    stim2_color = 'b' if stim[1] == 1 else 'r'
    return [stim1_color, stim2_color]

def get_trialtype_colors():
    stims = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    colors = ['#6E439A','#2B1644', '#236975','#49BEA3']
    return stims, colors

def get_fixation_baseline_times(times_dict):
    """
    Get the baseline period for the given times dictionary.
    """
    baseline = [int(times_dict['stim1_on']/2), int(times_dict['stim1_on'])]
    return baseline