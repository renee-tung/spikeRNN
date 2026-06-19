#!/usr/bin/env python3
"""Generate Sternberg task data from trained rate models.

This script is a notebook-to-script conversion of analysis/generate_rate_data_v2.ipynb,
with performance improvements focused on the heaviest path (trial generation/evaluation):

1) Parallelization across models with ProcessPoolExecutor.
2) Per-process model caching so .mat is loaded once per model worker.
3) Preallocation of output arrays instead of repeated list append.
4) Optional Torch backend for GPU execution (device=cuda) when available.

Examples
--------
python analysis/generate_rate_data_v2.py list-models \
    --model-dir /home/nuttidalab/Documents/renee/sternberg/interleaved_0.5

python analysis/generate_rate_data_v2.py generate-data \
    --model-dir /home/nuttidalab/Documents/renee/sternberg/interleaved_0.5 \
    --workers 8 --n-repetitions 3 --backend numpy

python analysis/generate_rate_data_v2.py generate-data \
    --model-dir /home/nuttidalab/Documents/renee/sternberg/interleaved_0.5 \
    --workers 1 --n-repetitions 3 --backend torch --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle as pk
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.stats import wilcoxon, kruskal

DEFAULT_SETTINGS: Dict[str, int | str] = {
    "T": 450,
    "stim_on": 50,
    "stim_dur": 25,
    # "delay": 200,
    "delay": 150,
    "DeltaT": 1,
    "taus": 20,
    "fs": 200,
    "task": "sternberg",
    "load": 1,
}


def find_files(path: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, pattern)))


def generate_input_stim_sternberg_stims(
    settings: Dict[str, Any],
    stim_ids: np.ndarray,
    max_load: int = 3,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Generate a Sternberg trial from explicit stimulus IDs.

    stim_ids format uses np.inf for inactive memory-set channels at lower loads.
    Final index (max_load) is probe channel.
    """
    t_total = int(settings["T"])
    stim_on = int(settings["stim_on"])
    stim_dur = int(settings["stim_dur"])
    delay = int(settings["delay"])

    load = int((~np.isinf(stim_ids)).sum() - 1)
    n_channels = 4

    u = np.zeros((n_channels, t_total), dtype=np.float32)

    ending_idx = stim_on
    for i, i_chan in enumerate(stim_ids[:load]):
        i_chan = int(i_chan)
        if i == 0:
            u[i_chan, stim_on : stim_on + stim_dur] = 1.0
            ending_idx = stim_on + stim_dur
        else:
            u[i_chan, ending_idx : ending_idx + stim_dur] = 1.0
            ending_idx = ending_idx + stim_dur

    probe = int(stim_ids[max_load])
    u[probe, ending_idx + delay : ending_idx + delay + stim_dur] = 1.0
    label = 1 if stim_ids[max_load] in stim_ids[:load] else -1
    return u, label, stim_ids


def generate_input_stim_sternberg_type(
    settings: Dict[str, Any],
    max_load: int = 3,
    match: int = 0,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Random Sternberg trial generator balancing match/mismatch by caller."""
    t_total = int(settings["T"])
    stim_on = int(settings["stim_on"])
    stim_dur = int(settings["stim_dur"])
    delay = int(settings["delay"])
    load = int(settings["load"])

    n_channels = 4
    stim_ids = np.full(max_load + 1, np.nan, dtype=np.float32)
    u = np.zeros((n_channels, t_total), dtype=np.float32)

    stim_chan_idx = np.random.choice(n_channels, load, replace=False)
    stim_ids[:load] = stim_chan_idx

    ending_idx = stim_on
    for i, i_chan in enumerate(stim_chan_idx):
        if i == 0:
            u[i_chan, stim_on : stim_on + stim_dur] = 1.0
            ending_idx = stim_on + stim_dur
        else:
            u[i_chan, ending_idx : ending_idx + stim_dur] = 1.0
            ending_idx = ending_idx + stim_dur

    if match == 1:
        match_chan_idx = int(np.random.choice(stim_chan_idx, 1)[0])
        u[match_chan_idx, ending_idx + delay : ending_idx + delay + stim_dur] = 1.0
        stim_ids[max_load] = match_chan_idx
        label = 1
    elif match == 0:
        non_stim_chan_idx = np.setdiff1d(np.arange(n_channels), stim_chan_idx)
        non_match_chan_idx = int(np.random.choice(non_stim_chan_idx, 1)[0])
        u[non_match_chan_idx, ending_idx + delay : ending_idx + delay + stim_dur] = 1.0
        stim_ids[max_load] = non_match_chan_idx
        label = -1
    else:
        raise ValueError("match must be 0 or 1")

    return u, label, stim_ids


def get_all_stim_combos() -> np.ndarray:
    """Generate all memory-set/probe combinations used in the notebook."""
    values = [0, 1, 2, 3]
    inf = np.inf

    load3 = np.array(
        [
            [a, b, c, probe]
            for a, b, c in permutations(values, 3)
            for probe in values
        ],
        dtype=float
    )

    load2 = np.array(
        [
            [a, b, inf, probe]
            for a, b in permutations(values, 2)
            for probe in values
        ],
        dtype=float
    )

    load1 = np.array(
        [
            [a, inf, inf, probe]
            for a, probe in product(values, repeat=2)
        ],
        dtype=float
    )

    # Combine all loads
    return np.vstack([load3, load2, load1])

@dataclass
class PreparedModel:
    n_units: int
    w: np.ndarray
    m: np.ndarray
    som_m: np.ndarray
    w_in: np.ndarray
    w_out: np.ndarray
    b_out: float
    taus_sig: np.ndarray
    exc_ind: np.ndarray


def _prepare_model_numpy(
    model_path: str,
    lesion: str = "",
    lesion_inds: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    lesion_scale: float = 0.5,
) -> PreparedModel:
    """Load and precompute model matrices once per worker/model."""
    var = scipy.io.loadmat(model_path)

    n_units = int(var["N"][0][0])
    w = var["w"].astype(np.float32)
    m = var["m"].astype(np.float32)
    som_m = var["som_m"].astype(np.float32)
    w_in = var["w_in"].astype(np.float32)
    w_out = var["w_out"].astype(np.float32)
    b_out = float(np.squeeze(var["b_out"]))

    taus_gaus = var["taus_gaus"]
    taus_minmax = var["taus"][0]
    taus_sig = (1.0 / (1.0 + np.exp(-taus_gaus)) * (taus_minmax[1] - taus_minmax[0])) + taus_minmax[0]
    taus_sig = np.squeeze(taus_sig).astype(np.float32)

    exc_ind = np.where(var["exc"] == 1)[0]
    inh_ind = np.where(var["inh"] == 1)[0]
    som_n = int(var["som_N"][0][0])

    if lesion:
        lesion_mask = np.ones_like(w, dtype=np.float32)
        if lesion == "ii":
            lesion_mask[np.ix_(inh_ind, inh_ind)] = lesion_scale
        elif lesion == "ei":
            lesion_mask[np.ix_(exc_ind, inh_ind)] = lesion_scale
        elif lesion == "ie":
            lesion_mask[np.ix_(inh_ind, exc_ind)] = lesion_scale
        elif lesion == "ee":
            lesion_mask[np.ix_(exc_ind, exc_ind)] = lesion_scale
        elif lesion == "custom" and lesion_inds is not None:
            for row_inds, col_inds in lesion_inds:
                if row_inds is None:
                    lesion_mask[:, col_inds] = lesion_scale
                elif col_inds is None:
                    lesion_mask[row_inds, :] = lesion_scale
                else:
                    lesion_mask[np.ix_(row_inds, col_inds)] = lesion_scale
        w = w * lesion_mask

    if som_n > 0:
        _ = inh_ind[:som_n]

    return PreparedModel(
        n_units=n_units,
        w=w,
        m=m,
        som_m=som_m,
        w_in=w_in,
        w_out=w_out,
        b_out=b_out,
        taus_sig=taus_sig,
        exc_ind=exc_ind,
    )


def eval_trial_numpy(
    prepared: PreparedModel,
    settings: Dict[str, Any],
    u: np.ndarray,
    calc_epsp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast numpy reimplementation of model.eval_tf with caching/precompute."""
    t_total = int(settings["T"])
    dt = float(settings["DeltaT"])

    n = prepared.n_units
    x = np.zeros((n, t_total), dtype=np.float32)
    r = np.zeros((n, t_total), dtype=np.float32)
    epsp = np.zeros((t_total,), dtype=np.float32)
    o = np.zeros((t_total,), dtype=np.float32)

    x[:, 0] = np.random.randn(n).astype(np.float32) / 100.0
    r[:, 0] = 1.0 / (1.0 + np.exp(-x[:, 0]))
    epsp[0] = np.mean(np.abs(np.random.randn(n).astype(np.float32) / 100.0))

    # Key speedup: invariant matrix product moved outside time loop.
    ww = (prepared.w @ prepared.m) * prepared.som_m
    alpha = (1.0 - dt / prepared.taus_sig).astype(np.float32)
    beta = (dt / prepared.taus_sig).astype(np.float32)

    for t in range(1, t_total):
        rec_term = ww @ r[:, t - 1]
        in_term = prepared.w_in @ u[:, t - 1]
        next_x = alpha * x[:, t - 1] + beta * (rec_term + in_term) + np.random.randn(n).astype(np.float32) / 10.0

        if calc_epsp:
            next_epsp = alpha * x[:, t - 1] + beta * (ww[:, prepared.exc_ind] @ r[prepared.exc_ind, t - 1])
            epsp[t] = np.mean(next_epsp)

        x[:, t] = next_x
        r[:, t] = 1.0 / (1.0 + np.exp(-next_x))
        raw_output = prepared.w_out @ r[:, t] + prepared.b_out
        o[t] = np.asarray(raw_output).reshape(-1)[0]

    return x, r, o, epsp


def eval_trial_torch(
    prepared: PreparedModel,
    settings: Dict[str, Any],
    u: np.ndarray,
    device: str = "cuda",
    calc_epsp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Torch backend for optional GPU acceleration.

    Note: Keep workers=1 for GPU mode to avoid memory contention.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable") from exc

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device if use_cuda else "cpu")

    t_total = int(settings["T"])
    dt = float(settings["DeltaT"])

    n = prepared.n_units
    x = torch.zeros((n, t_total), dtype=torch.float32, device=dev)
    r = torch.zeros((n, t_total), dtype=torch.float32, device=dev)
    epsp = torch.zeros((t_total,), dtype=torch.float32, device=dev)
    o = torch.zeros((t_total,), dtype=torch.float32, device=dev)

    w = torch.from_numpy(prepared.w).to(dev)
    m = torch.from_numpy(prepared.m).to(dev)
    som_m = torch.from_numpy(prepared.som_m).to(dev)
    w_in = torch.from_numpy(prepared.w_in).to(dev)
    w_out = torch.from_numpy(prepared.w_out).to(dev)
    taus_sig = torch.from_numpy(prepared.taus_sig).to(dev)
    u_t = torch.from_numpy(u).to(dev)

    x[:, 0] = torch.randn(n, device=dev) / 100.0
    r[:, 0] = torch.sigmoid(x[:, 0])
    epsp[0] = torch.mean(torch.abs(torch.randn(n, device=dev) / 100.0))

    ww = (w @ m) * som_m
    alpha = 1.0 - dt / taus_sig
    beta = dt / taus_sig
    exc_ind = torch.from_numpy(prepared.exc_ind).to(dev)

    for t in range(1, t_total):
        rec_term = ww @ r[:, t - 1]
        in_term = w_in @ u_t[:, t - 1]
        next_x = alpha * x[:, t - 1] + beta * (rec_term + in_term) + torch.randn(n, device=dev) / 10.0

        if calc_epsp:
            next_epsp = alpha * x[:, t - 1] + beta * (ww[:, exc_ind] @ r[exc_ind, t - 1])
            epsp[t] = torch.mean(next_epsp)

        x[:, t] = next_x
        r[:, t] = torch.sigmoid(next_x)
        o[t] = torch.matmul(w_out, r[:, t]) + prepared.b_out

    return (
        x.detach().cpu().numpy(),
        r.detach().cpu().numpy(),
        o.detach().cpu().numpy(),
        epsp.detach().cpu().numpy(),
    )


def _run_single_model_worker_balance_match(
    model_path: str,
    model_dir: str,
    settings: Dict[str, Any],
    n_trials: int,
    loads: List[int],
    eval_amp_threshold: float,
    backend: str,
    device: str,
    overwrite: bool,
    lesion: str = "",
    lesion_inds: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    lesion_scale: float = 0.5,
    lesion_tag: str = "",
) -> str:
    """Worker for random trial generation with balanced match/mismatch per load."""
    model_name = os.path.basename(model_path)
    output_dir = os.path.join(model_dir, model_name[:-4])
    os.makedirs(output_dir, exist_ok=True)

    lesion_suffix = f"_lesion_{lesion_tag or lesion}" if lesion else ""
    neural_path = os.path.join(output_dir, f"neural_dict_delay{settings['delay']}_balance_match{lesion_suffix}.pkl")
    bhv_path = os.path.join(output_dir, f"bhv_dict_delay{settings['delay']}_balance_match{lesion_suffix}.pkl")
    settings_path = os.path.join(output_dir, f"settings_dict_delay{settings['delay']}_balance_match{lesion_suffix}.pkl")

    if (not overwrite) and os.path.exists(neural_path) and os.path.exists(bhv_path):
        return f"skipping {model_name}, already saved"

    prepared = _prepare_model_numpy(model_path, lesion=lesion, lesion_inds=lesion_inds, lesion_scale=lesion_scale)

    # Pre-compute max_load from loads
    max_load = max(loads)
    n_total = len(loads) * 2 * n_trials  # 2 match conditions per load, n_trials each

    # Probe first trial to allocate output arrays
    local_settings = settings.copy()
    local_settings["load"] = loads[0]
    first_u, first_label, first_stim_ids = generate_input_stim_sternberg_type(
        local_settings, max_load=max_load, match=0
    )
    if backend == "torch":
        _, first_r, first_o, first_epsp = eval_trial_torch(prepared, settings, first_u, device=device, calc_epsp=True)
    else:
        _, first_r, first_o, first_epsp = eval_trial_numpy(prepared, settings, first_u, calc_epsp=True)

    r_all = np.empty((n_total,) + first_r.shape, dtype=first_r.dtype)
    epsp_all = np.empty((n_total,) + np.atleast_1d(first_epsp).shape, dtype=np.atleast_1d(first_epsp).dtype)
    out_all = np.empty((n_total,) + first_o.shape, dtype=first_o.dtype)
    perf_all = np.empty((n_total,), dtype=np.int8)
    trial_type_all = np.empty((n_total, 2 + first_stim_ids.size), dtype=np.float32)

    idx = 0
    for load in loads:
        local_settings = settings.copy()
        local_settings["load"] = load
        resp_onset = int(
            local_settings["stim_on"] + load * local_settings["stim_dur"] + local_settings["delay"] + 10
        )

        # Generate n_trials for match=0 and n_trials for match=1 (balanced)
        for match in [0, 1]:
            for _ in range(n_trials):
                u, label, stim_ids = generate_input_stim_sternberg_type(
                    local_settings, max_load=max_load, match=match
                )

                if backend == "torch":
                    _, r, o, epsp = eval_trial_torch(prepared, settings, u, device=device, calc_epsp=True)
                else:
                    _, r, o, epsp = eval_trial_numpy(prepared, settings, u, calc_epsp=True)

                if label == 1:
                    perf = int(np.max(o[resp_onset:]) > eval_amp_threshold)
                else:
                    perf = int(np.min(o[resp_onset:]) < -eval_amp_threshold)

                r_all[idx] = r
                epsp_all[idx] = epsp
                out_all[idx] = o
                perf_all[idx] = perf
                trial_type_all[idx] = np.concatenate((np.array([load, label], dtype=np.float32), stim_ids.flatten()))
                idx += 1

    neural_dict = {"r": r_all, "epsp": epsp_all}
    bhv_dict = {"perf": perf_all, "trial_type": trial_type_all, "outputs": out_all}

    with open(neural_path, "wb") as f:
        pk.dump(neural_dict, f)
    with open(bhv_path, "wb") as f:
        pk.dump(bhv_dict, f)
    with open(settings_path, "wb") as f:
        pk.dump(settings.copy(), f)

    return f"saved model outputs to {output_dir}"


def _run_single_model_worker(
    model_path: str,
    model_dir: str,
    settings: Dict[str, Any],
    trial_stims: np.ndarray,
    n_repetitions: int,
    eval_amp_threshold: float,
    backend: str,
    device: str,
    overwrite: bool,
    lesion: str = "",
    lesion_inds: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    lesion_scale: float = 0.5,
    lesion_tag: str = "",
) -> str:
    model_name = os.path.basename(model_path)
    output_dir = os.path.join(model_dir, model_name[:-4])
    os.makedirs(output_dir, exist_ok=True)

    lesion_suffix = f"_lesion_{lesion_tag or lesion}" if lesion else ""
    neural_path = os.path.join(output_dir, f"neural_dict_delay{settings['delay']}_balance_stims{lesion_suffix}.pkl")
    bhv_path = os.path.join(output_dir, f"bhv_dict_delay{settings['delay']}_balance_stims{lesion_suffix}.pkl")
    settings_path = os.path.join(output_dir, f"settings_dict_delay{settings['delay']}_balance_stims{lesion_suffix}.pkl")

    if (not overwrite) and os.path.exists(neural_path) and os.path.exists(bhv_path):
        return f"skipping {model_name}, already saved"

    prepared = _prepare_model_numpy(model_path, lesion=lesion, lesion_inds=lesion_inds, lesion_scale=lesion_scale)

    max_load = trial_stims.shape[1] - 1
    trial_specs: List[Tuple[np.ndarray, int, int, np.ndarray, int]] = []

    for stim_ids in trial_stims:
        load = int((~np.isinf(stim_ids)).sum() - 1)
        local_settings = settings.copy()
        local_settings["load"] = load
        u, label, stim_ids_used = generate_input_stim_sternberg_stims(local_settings, stim_ids, max_load=max_load)
        resp_onset = int(local_settings["stim_on"] + load * local_settings["stim_dur"] + local_settings["delay"] + 10)
        trial_specs.append((u, label, load, np.array(stim_ids_used, dtype=np.float32), resp_onset))

    n_total = len(trial_specs) * n_repetitions
    if n_total == 0:
        raise RuntimeError(f"No trial specs built for {model_name}")

    # Probe first trial to allocate output arrays with exact shape/dtype.
    first_u, first_label, first_load, first_stim_ids, first_resp = trial_specs[0]
    if backend == "torch":
        _, first_r, first_o, first_epsp = eval_trial_torch(prepared, settings, first_u, device=device, calc_epsp=True)
    else:
        _, first_r, first_o, first_epsp = eval_trial_numpy(prepared, settings, first_u, calc_epsp=True)

    r_all = np.empty((n_total,) + first_r.shape, dtype=first_r.dtype)
    epsp_all = np.empty((n_total,) + np.atleast_1d(first_epsp).shape, dtype=np.atleast_1d(first_epsp).dtype)
    out_all = np.empty((n_total,) + first_o.shape, dtype=first_o.dtype)
    perf_all = np.empty((n_total,), dtype=np.int8)
    trial_type_all = np.empty((n_total, 2 + first_stim_ids.size), dtype=np.float32)

    idx = 0
    for (u, label, load, stim_ids_used, resp_onset) in trial_specs:
        for _ in range(n_repetitions):
            if backend == "torch":
                _, r, o, epsp = eval_trial_torch(prepared, settings, u, device=device, calc_epsp=True)
            else:
                _, r, o, epsp = eval_trial_numpy(prepared, settings, u, calc_epsp=True)

            if label == 1:
                perf = int(np.max(o[resp_onset:]) > eval_amp_threshold)
            else:
                perf = int(np.min(o[resp_onset:]) < -eval_amp_threshold)

            r_all[idx] = r
            epsp_all[idx] = epsp
            out_all[idx] = o
            perf_all[idx] = perf
            trial_type_all[idx] = np.concatenate((np.array([load, label], dtype=np.float32), stim_ids_used.flatten()))
            idx += 1

    neural_dict = {"r": r_all, "epsp": epsp_all}
    bhv_dict = {"perf": perf_all, "trial_type": trial_type_all, "outputs": out_all}

    with open(neural_path, "wb") as f:
        pk.dump(neural_dict, f)
    with open(bhv_path, "wb") as f:
        pk.dump(bhv_dict, f)
    with open(settings_path, "wb") as f:
        pk.dump(settings.copy(), f)

    return f"saved model outputs to {output_dir}"


def _run_single_model_worker_precomputed(
    model_path: str,
    model_dir: str,
    settings: Dict[str, Any],
    u_all: np.ndarray,
    trial_type_all: np.ndarray,
    eval_amp_threshold: float,
    backend: str,
    device: str,
    overwrite: bool,
    lesion: str = "",
    lesion_inds: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    lesion_scale: float = 0.5,
    lesion_tag: str = "",
) -> str:
    """Worker that evaluates pre-generated stimuli passed in as u_all.

    u_all : (n_trials, n_channels, t_total)
    trial_type_all : (n_trials, ...) — saved directly into bhv_dict.
        Columns 0 and 1 are expected to be load and label (1 / -1) so that
        performance can be computed; other columns are passed through as-is.
    """
    model_name = os.path.basename(model_path)
    output_dir = os.path.join(model_dir, model_name[:-4])
    os.makedirs(output_dir, exist_ok=True)

    lesion_suffix = f"_lesion_{lesion_tag or lesion}" if lesion else ""
    neural_path = os.path.join(output_dir, f"neural_dict_delay{settings['delay']}_precomputed{lesion_suffix}.pkl")
    bhv_path = os.path.join(output_dir, f"bhv_dict_delay{settings['delay']}_precomputed{lesion_suffix}.pkl")
    settings_path = os.path.join(output_dir, f"settings_dict_delay{settings['delay']}_precomputed{lesion_suffix}.pkl")

    if (not overwrite) and os.path.exists(neural_path) and os.path.exists(bhv_path):
        return f"skipping {model_name}, already saved"

    prepared = _prepare_model_numpy(model_path, lesion=lesion, lesion_inds=lesion_inds, lesion_scale=lesion_scale)

    n_total = len(u_all)

    first_u = u_all[0]
    if backend == "torch":
        _, first_r, first_o, first_epsp = eval_trial_torch(prepared, settings, first_u, device=device, calc_epsp=True)
    else:
        _, first_r, first_o, first_epsp = eval_trial_numpy(prepared, settings, first_u, calc_epsp=True)

    r_all = np.empty((n_total,) + first_r.shape, dtype=first_r.dtype)
    epsp_all = np.empty((n_total,) + np.atleast_1d(first_epsp).shape, dtype=np.atleast_1d(first_epsp).dtype)
    out_all = np.empty((n_total,) + first_o.shape, dtype=first_o.dtype)
    perf_all = np.zeros(n_total, dtype=np.int8)

    def _compute_perf(o, trial_type_row):
        load = int(trial_type_row[0])
        label = int(trial_type_row[1])
        resp_onset = int(settings["stim_on"] + load * settings["stim_dur"] + settings["delay"] + 10)
        if label == 1:
            return int(np.max(o[resp_onset:]) > eval_amp_threshold)
        else:
            return int(np.min(o[resp_onset:]) < -eval_amp_threshold)

    r_all[0] = first_r
    epsp_all[0] = first_epsp
    out_all[0] = first_o
    perf_all[0] = _compute_perf(first_o, trial_type_all[0])

    for i in range(1, n_total):
        if backend == "torch":
            _, r, o, epsp = eval_trial_torch(prepared, settings, u_all[i], device=device, calc_epsp=True)
        else:
            _, r, o, epsp = eval_trial_numpy(prepared, settings, u_all[i], calc_epsp=True)

        r_all[i] = r
        epsp_all[i] = epsp
        out_all[i] = o
        perf_all[i] = _compute_perf(o, trial_type_all[i])

    neural_dict = {"r": r_all, "epsp": epsp_all}
    bhv_dict = {"perf": perf_all, "trial_type": trial_type_all, "outputs": out_all}

    with open(neural_path, "wb") as f:
        pk.dump(neural_dict, f)
    with open(bhv_path, "wb") as f:
        pk.dump(bhv_dict, f)
    with open(settings_path, "wb") as f:
        pk.dump(settings.copy(), f)

    return f"saved model outputs to {output_dir}"


def list_models(model_dir: str, pattern: str = "*N_1000*.mat") -> List[str]:
    model_names = find_files(model_dir, pattern)
    print(len(model_names))
    for name in model_names:
        print(name)
    return model_names


def move_bad_training_models(model_dir: str, pattern: str = "*N_1000*.mat") -> None:
    model_names = find_files(model_dir, pattern)
    new_dir = os.path.join(model_dir, "old")
    os.makedirs(new_dir, exist_ok=True)

    for model_path in model_names:
        model_data = scipy.io.loadmat(model_path)
        tr_trials = int(model_data["tr"][0][0])
        if tr_trials == 39999:
            model_name_nopath = os.path.basename(model_path)
            print(f"moving {model_name_nopath} with {tr_trials} trials")
            os.rename(model_path, os.path.join(new_dir, model_name_nopath))


def print_model_performance(model_dir: str, pattern: str = "*N_1000*.mat") -> None:
    model_names = find_files(model_dir, pattern)
    for model_path in model_names:
        model_name = os.path.basename(model_path)
        model_data = scipy.io.loadmat(model_path)
        perfs = model_data["eval_perfs"][0]
        train_trials = int(model_data["tr"][0][0])
        print(f"Model: {model_name}, Test perf: {perfs}, Train trials: {train_trials}")


def generate_data(
    model_dir: str,
    pattern: str,
    n_repetitions: int,
    eval_amp_threshold: float,
    workers: int,
    backend: str,
    device: str,
    overwrite: bool,
    mode: str = "balance_stims",
    n_trials: int = 50,
    loads: List[int] = None,
    delay: int = None,
    lesion: str = "",
    lesion_inds: Optional[List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    lesion_scale: float = 0.5,
    lesion_tag: str = "",
    include_models: Optional[Sequence[str]] = None,
    u_all: Optional[np.ndarray] = None,
    trial_type_all: Optional[np.ndarray] = None,
) -> None:
    """Generate trial data for Sternberg task models.

    Parameters
    ----------
    mode : str
        "balance_stims": Use all combinations from get_all_stim_combos() repeated n_repetitions times.
        "balance_match": Randomly generate trials with balanced match/mismatch per load.
        "precomputed": Evaluate pre-generated stimuli supplied via u_all / trial_type_all.
    n_trials : int
        Number of trials per load per match condition (used in balance_match mode).
    loads : List[int]
        List of load values to use (used in balance_match mode). Default [1, 3].
    u_all : np.ndarray, shape (n_trials, n_channels, t_total)
        Pre-generated input stimuli (precomputed mode only).
    trial_type_all : np.ndarray, shape (n_trials, ...)
        Trial metadata saved into bhv_dict (precomputed mode only).
        Column 0 = load, column 1 = label (1/-1) are used to compute performance.
    """
    if loads is None:
        loads = [1, 2, 3]

    model_list = _select_model_paths(model_dir, pattern, include_models)
    if not model_list:
        print("No models found.")
        return

    settings = DEFAULT_SETTINGS.copy()
    if delay is not None:
        settings["delay"] = delay

    if backend == "torch" and device.startswith("cuda") and workers > 1:
        print("GPU mode with multiple workers can cause contention; forcing workers=1.")
        workers = 1

    total_models = len(model_list)
    print(
        f"Starting generate-data: mode={mode}, models={total_models}, workers={workers}, backend={backend}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        if mode == "balance_stims":
            trial_stims = get_all_stim_combos()
            futures = [
                executor.submit(
                    _run_single_model_worker,
                    model_path,
                    model_dir,
                    settings,
                    trial_stims,
                    n_repetitions,
                    eval_amp_threshold,
                    backend,
                    device,
                    overwrite,
                    lesion,
                    lesion_inds,
                    lesion_scale,
                    lesion_tag,
                )
                for model_path in model_list
            ]
        elif mode == "balance_match":
            futures = [
                executor.submit(
                    _run_single_model_worker_balance_match,
                    model_path,
                    model_dir,
                    settings,
                    n_trials,
                    loads,
                    eval_amp_threshold,
                    backend,
                    device,
                    overwrite,
                    lesion,
                    lesion_inds,
                    lesion_scale,
                    lesion_tag,
                )
                for model_path in model_list
            ]
        elif mode == "precomputed":
            if u_all is None or trial_type_all is None:
                raise ValueError("mode='precomputed' requires u_all and trial_type_all to be provided.")
            futures = [
                executor.submit(
                    _run_single_model_worker_precomputed,
                    model_path,
                    model_dir,
                    settings,
                    u_all,
                    trial_type_all,
                    eval_amp_threshold,
                    backend,
                    device,
                    overwrite,
                    lesion,
                    lesion_inds,
                    lesion_scale,
                    lesion_tag,
                )
                for model_path in model_list
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'balance_stims', 'balance_match', or 'precomputed'.")

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result_msg = future.result()
                print(f"[{i}/{total_models}] {result_msg}", flush=True)
            except Exception as exc:
                print(f"[{i}/{total_models}] worker failed: {exc}", flush=True)
                raise

def _normalize_model_name(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".mat"):
        return base
    return f"{base}.mat"


def _select_model_paths(
    model_dir: str,
    pattern: str,
    include_models: Sequence[str] | None = None,
    delay: int = 200,
) -> List[str]:
    all_paths = find_files(model_dir, pattern)
    if include_models is None:
        return all_paths

    wanted = {_normalize_model_name(name) for name in include_models}
    selected = [p for p in all_paths if os.path.basename(p) in wanted]
    selected_names = {os.path.basename(p) for p in selected}
    missing = sorted(wanted - selected_names)
    if missing:
        print(f"Warning: {len(missing)} requested model(s) not found: {missing}")
    return selected


def _filter_models_by_delay(
    model_paths: List[str],
    model_dir: str,
    delay: int,
    mode: str,
) -> List[str]:
    """Return only model paths that have generated data files for the given delay and mode."""
    filtered = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(model_dir, model_name[:-4])
        candidates = [
            os.path.join(output_dir, f"bhv_dict_delay{delay}_{mode}.pkl"),
            os.path.join(output_dir, f"bhv_dict_delay{delay}.pkl"),
        ]
        if any(os.path.exists(p) for p in candidates):
            filtered.append(model_path)
    n_dropped = len(model_paths) - len(filtered)
    if n_dropped:
        print(f"Filtered out {n_dropped} model(s) with no data for delay={delay}, mode={mode}")
    return filtered


def _load_behavior_artifacts(
    output_dir: str,
    mode: str = "balance_stims",
    default_delay: int = 200,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if mode not in {"balance_stims", "balance_match"}:
        raise ValueError(f"Unknown mode: {mode}")

    settings_candidates = [
        os.path.join(output_dir, f"settings_dict_delay{default_delay}_{mode}.pkl"),
        os.path.join(output_dir, f"settings_dict_delay{default_delay}.pkl"),
    ]

    settings_path = next((p for p in settings_candidates if os.path.exists(p)), None)
    if settings_path is None:
        raise FileNotFoundError(f"settings file not found for mode={mode} in {output_dir}")

    with open(settings_path, "rb") as f:
        settings = pk.load(f)

    delay_val = int(settings.get("delay", default_delay))
    bhv_candidates = [
        os.path.join(output_dir, f"bhv_dict_delay{delay_val}_{mode}.pkl"),
        os.path.join(output_dir, f"bhv_dict_delay{delay_val}.pkl"),
    ]

    bhv_path = next((p for p in bhv_candidates if os.path.exists(p)), None)
    if bhv_path is None:
        raise FileNotFoundError(f"behavior file not found for mode={mode} in {output_dir}")

    with open(bhv_path, "rb") as f:
        bhv_dict = pk.load(f)

    return settings, bhv_dict


def compute_delay200_perfs(
    model_dir: str,
    pattern: str = "*N_1000*.mat",
    mode: str = "balance_stims",
    include_models: Sequence[str] | None = None,
    delay: int = 200,
) -> np.ndarray:
    model_paths = _select_model_paths(model_dir, pattern, include_models)
    loads = [1, 2, 3]
    test_perfs = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(model_dir, model_name[:-4])

        try:
            settings, bhv_dict = _load_behavior_artifacts(output_dir, mode=mode, default_delay=delay)
        except FileNotFoundError:
            continue

        trial_type = np.array(bhv_dict["trial_type"])
        perfs = np.array(bhv_dict["perf"])

        load_perfs = np.zeros(3, dtype=np.float32)
        for n, load in enumerate(loads):
            this_load_idx = np.where(trial_type[:, 0] == load)[0]
            load_perfs[n] = np.mean(perfs[this_load_idx])
        test_perfs.append(load_perfs)

    return np.array(test_perfs)


def plot_quick_behavior(
    model_dir: str,
    pattern: str = "*N_1000*.mat",
    mode: str = "balance_stims",
    include_models: Sequence[str] | None = None,
    delay: int = 200,
) -> None:
    model_paths = _select_model_paths(model_dir, pattern, include_models, delay=delay)
    if not model_paths:
        print("No models matched selection.")
        return

    colors = ["#49BEA3", "#6E439A"]
    settings = DEFAULT_SETTINGS.copy()

    for load in [1, 2, 3]:
        n_models = len(model_paths)
        n_cols = 5
        n_rows = int(np.ceil(n_models / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axs = np.atleast_1d(axs).flatten()
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            output_dir = os.path.join(model_dir, model_name[:-4])

            try:
                settings, bhv_dict = _load_behavior_artifacts(output_dir, mode=mode, default_delay=delay)
            except Exception:
                print(f"No data for {model_name}, skipping")
                continue

            trial_type = np.array(bhv_dict["trial_type"])
            this_load_idx = np.where(trial_type[:, 0] == load)[0]
            for n in this_load_idx:
                axs[i].plot(
                    bhv_dict["outputs"][n],
                    color=colors[int(trial_type[n][1] == 1)],
                    alpha=0.3,
                )

            perf = np.mean(np.array(bhv_dict["perf"])[this_load_idx])
            axs[i].set_title(f"{model_name[-10:-4]}, Perf: {perf:.2f}")
            for n in range(load):
                stim_on = settings["stim_on"] + n * settings["stim_dur"]
                axs[i].axvline(stim_on, color="gray", alpha=0.3)
            axs[i].axvspan(
                stim_on + settings["delay"],
                stim_on + settings["delay"] + settings["stim_dur"],
                color="gray",
                alpha=0.3,
            )

        for j in range(len(model_paths), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()


def plot_behavior_boxplots(
    model_dir: str,
    pattern: str = "*N_1000*.mat",
    mode: str = "balance_stims",
    include_models: Sequence[str] | None = None,
    delay: int = 200,
) -> None:
    model_paths = _select_model_paths(model_dir, pattern, include_models)
    if not model_paths:
        print("No models matched selection.")
        return

    perfs = []
    for model_path in model_paths:
        model_data = scipy.io.loadmat(model_path)
        perfs.append(model_data["eval_perfs"][0])

    perfs = np.array(perfs)
    test_perfs = compute_delay200_perfs(model_dir, pattern, mode=mode, include_models=include_models, delay=delay)
    if test_perfs.size == 0:
        print("No test performance files found for selected mode/models.")
        return
    n_common = min(perfs.shape[0], test_perfs.shape[0])
    perfs = perfs[:n_common]
    test_perfs = test_perfs[:n_common]
    loads = [1, 2, 3]

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    axs = axs.flatten()

    for i, load in enumerate(loads):
        axs[i].boxplot([perfs[:, i], test_perfs[:, i]], labels=["Delay=50", f"Delay={delay}"])
        axs[i].set_title(f"Load {load}")
        axs[i].set_ylabel("Performance")

        stat, p = wilcoxon(perfs[:, i], test_perfs[:, i])
        _ = stat
        axs[i].text(0.5, 0.95, f"p={p:.3f}", transform=axs[i].transAxes, ha="center", va="top")

        axs[i].axhline(0.5, color="red", linestyle="--", alpha=0.5)
        for j in range(perfs.shape[0]):
            axs[i].plot([1, 2], [perfs[j, i], test_perfs[j, i]], color="gray", alpha=0.3)

    axs[3].boxplot([perfs[:, 0], perfs[:, 1], perfs[:, 2]], labels=["Load 1", "Load 2", "Load 3"])
    axs[3].set_title("Training (delay=50) Performance by Load")
    axs[3].set_ylabel("Performance")
    stat, p = kruskal(perfs[:, 0], perfs[:, 1], perfs[:, 2])
    _ = stat
    axs[3].text(0.5, 0.95, f"p={p:.3f}", transform=axs[3].transAxes, ha="center", va="top")
    axs[3].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    for j in range(perfs.shape[0]):
        axs[3].plot([1, 2, 3], [perfs[j, 0], perfs[j, 1], perfs[j, 2]], color="gray", alpha=0.3)

    axs[4].boxplot([test_perfs[:, 0], test_perfs[:, 1], test_perfs[:, 2]], labels=["Load 1", "Load 2", "Load 3"])
    axs[4].set_title(f"Test (delay={delay}) Performance by Load")
    axs[4].set_ylabel("Performance")
    stat, p = kruskal(test_perfs[:, 0], test_perfs[:, 1], test_perfs[:, 2])
    _ = stat
    axs[4].text(0.5, 0.95, f"p={p:.3f}", transform=axs[4].transAxes, ha="center", va="top")
    for j in range(test_perfs.shape[0]):
        axs[4].plot([1, 2, 3], [test_perfs[j, 0], test_perfs[j, 1], test_perfs[j, 2]], color="gray", alpha=0.3)
    axs[4].axhline(0.5, color="red", linestyle="--", alpha=0.5)
    axs[4].axhline(0.7, color="green", linestyle="--", alpha=0.5)
    axs[4].axhline(0.6, color="yellow", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def move_low_perf_models(
    model_dir: str,
    threshold: float = 0.7,
    pattern: str = "*N_1000*.mat",
    mode: str = "balance_stims",
    actually_move: bool = False,
    delay: int = 200,
) -> None:
    model_names = find_files(model_dir, pattern)
    new_dir = os.path.join(model_dir, "low_perf")
    os.makedirs(new_dir, exist_ok=True)
    loads = [1, 2, 3]

    low_perf_models = []
    for model_path in model_names:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(model_dir, model_name[:-4])

        try:
            with open(os.path.join(output_dir, f"settings_dict_delay{delay}_{mode}.pkl"), "rb") as f:
                settings = pk.load(f)
            with open(os.path.join(output_dir, f"bhv_dict_delay{delay}_{mode}.pkl"), "rb") as f:
                bhv_dict = pk.load(f)
        except Exception:
            print(f"No data for {model_name}, skipping")
            continue

        trial_type = np.array(bhv_dict["trial_type"])
        perfs = np.array(bhv_dict["perf"])

        load_perfs = np.zeros(3, dtype=np.float32)
        for n, load in enumerate(loads):
            this_load_idx = np.where(trial_type[:, 0] == load)[0]
            load_perfs[n] = np.mean(perfs[this_load_idx])

        if np.any(load_perfs < threshold):
            print(f"Moving {model_name} with delay={delay} perf {load_perfs} to low_perf folder")
            low_perf_models.append(model_name)
            if actually_move:
                # Move the model .mat file
                os.rename(model_path, os.path.join(new_dir, model_name))
                # Move the output folder (same name as model without .mat extension)
                output_folder_name = model_name[:-4]
                output_folder_path = os.path.join(model_dir, output_folder_name)
                if os.path.exists(output_folder_path):
                    os.rename(output_folder_path, os.path.join(new_dir, output_folder_name))
                    
    # return list of model names with low perf models removed
    model_names = [os.path.basename(p) for p in model_names]
    return [name for name in model_names if name not in low_perf_models]


def rename_old_files_to_balance_stims(model_dir: str, pattern: str = "*N_1000*.mat") -> None:
    """Rename old saved files (without mode suffix) to include _balance_stims suffix.
    
    This is useful for migrating files saved with the old naming scheme.
    Files already having a mode suffix (balance_stims, balance_match) are skipped.
    """
    model_names = find_files(model_dir, pattern)
    renamed_count = 0
    skipped_count = 0

    for model_path in model_names:
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(model_dir, model_name[:-4])
        
        if not os.path.exists(output_dir):
            continue

        # Find files matching old naming scheme (delay200.pkl without mode suffix)
        for delay_val in [50, 200]:
            old_neural = os.path.join(output_dir, f"neural_dict_delay{delay_val}.pkl")
            old_bhv = os.path.join(output_dir, f"bhv_dict_delay{delay_val}.pkl")
            old_settings = os.path.join(output_dir, f"settings_dict_delay{delay_val}.pkl")
            
            new_neural = os.path.join(output_dir, f"neural_dict_delay{delay_val}_balance_stims.pkl")
            new_bhv = os.path.join(output_dir, f"bhv_dict_delay{delay_val}_balance_stims.pkl")
            new_settings = os.path.join(output_dir, f"settings_dict_delay{delay_val}_balance_stims.pkl")

            # Only rename if old files exist and new files don't
            if os.path.exists(old_neural) and not os.path.exists(new_neural):
                os.rename(old_neural, new_neural)
                print(f"Renamed: {os.path.basename(old_neural)} → {os.path.basename(new_neural)}")
                renamed_count += 1
            elif os.path.exists(new_neural):
                skipped_count += 1
            
            if os.path.exists(old_bhv) and not os.path.exists(new_bhv):
                os.rename(old_bhv, new_bhv)
                print(f"Renamed: {os.path.basename(old_bhv)} → {os.path.basename(new_bhv)}")
                renamed_count += 1
            elif os.path.exists(new_bhv):
                skipped_count += 1
            
            if os.path.exists(old_settings) and not os.path.exists(new_settings):
                os.rename(old_settings, new_settings)
                print(f"Renamed: {os.path.basename(old_settings)} → {os.path.basename(new_settings)}")
                renamed_count += 1
            elif os.path.exists(new_settings):
                skipped_count += 1

    print(f"\nRename complete: {renamed_count} files renamed, {skipped_count} files already named correctly or skipped.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and analyze Sternberg rate-model outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model-dir",
        type=str,
        default="/home/nuttidalab/Documents/renee/sternberg/interleaved_0.5",
        help="Directory containing model .mat files",
    )
    common.add_argument("--pattern", type=str, default="*N_1000*.mat", help="Model filename glob pattern")

    subparsers.add_parser("list-models", parents=[common])
    subparsers.add_parser("move-bad-training", parents=[common])
    subparsers.add_parser("print-model-perf", parents=[common])
    plot_quick = subparsers.add_parser("plot-quick-behavior", parents=[common])
    plot_quick.add_argument("--mode", choices=["balance_stims", "balance_match"], default="balance_stims")
    plot_quick.add_argument("--delay", type=int, default=200, help="Delay duration used in file names (e.g. 150 loads neural_dict_delay150_*.pkl)")
    plot_quick.add_argument("--models", nargs="+", default=None,
                            help="Optional model names (with or without .mat) to include")

    plot_box = subparsers.add_parser("plot-boxplots", parents=[common])
    plot_box.add_argument("--mode", choices=["balance_stims", "balance_match"], default="balance_stims")
    plot_box.add_argument("--delay", type=int, default=200, help="Delay duration used in file names")
    plot_box.add_argument("--models", nargs="+", default=None,
                          help="Optional model names (with or without .mat) to include")
    subparsers.add_parser("rename-files", parents=[common], help="Rename old saved files to include _balance_stims suffix")

    move_low = subparsers.add_parser("move-low-perf", parents=[common])
    move_low.add_argument("--threshold", type=float, default=0.7)
    move_low.add_argument("--actually-move", action="store_true", help="Perform file moves")

    gen = subparsers.add_parser("generate-data", parents=[common])
    gen.add_argument("--mode", choices=["balance_stims", "balance_match", "precomputed"], default="balance_stims",
                     help="Trial generation mode: balance_stims (all stimulus combos), balance_match (balanced match/mismatch per load), or precomputed (supply u_all via notebook only)")
    gen.add_argument("--n-repetitions", type=int, default=3,
                     help="Number of repetitions per stimulus combo (balance_stims mode only)")
    gen.add_argument("--n-trials", type=int, default=50,
                     help="Number of trials per load per match condition (balance_match mode only)")
    gen.add_argument("--loads", type=int, nargs="+", default=[1, 2, 3],
                     help="Load values to use (balance_match mode only)")
    gen.add_argument("--delay", type=int, default=None,
                     help="Delay duration in timesteps (overrides DEFAULT_SETTINGS; e.g. 150 → neural_dict_delay150_*.pkl)")
    gen.add_argument("--eval-amp-threshold", type=float, default=0.7)
    gen.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    gen.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    gen.add_argument("--device", type=str, default="cuda")
    gen.add_argument("--overwrite", action="store_true")
    gen.add_argument("--models", nargs="+", default=None,
                     help="Optional model names (with or without .mat) to run; runs all if omitted")
    gen.add_argument(
        "--lesion",
        type=str,
        default="",
        choices=["", "ee", "ei", "ie", "ii", "custom"],
        help="Lesion type. Use 'custom' with --lesion-inds-file to specify arbitrary indices.",
    )
    gen.add_argument(
        "--lesion-inds-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file defining custom lesion index pairs (used when --lesion=custom). "
            "Format: [[row_list_or_null, col_list_or_null], ...] — null means all rows/cols."
        ),
    )
    gen.add_argument(
        "--lesion-scale",
        type=float,
        default=0.5,
        help="Multiplicative scale applied to lesioned weights (default 0.5; use 0.0 for full ablation).",
    )
    gen.add_argument(
        "--lesion-tag",
        type=str,
        default="",
        help="Custom string used in the output filename suffix (e.g. 'tuned_ee'). Defaults to the lesion type.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "list-models":
        list_models(args.model_dir, args.pattern)
    elif args.command == "move-bad-training":
        move_bad_training_models(args.model_dir, args.pattern)
    elif args.command == "print-model-perf":
        print_model_performance(args.model_dir, args.pattern)
    elif args.command == "generate-data":
        lesion_inds = None
        if args.lesion == "custom" and args.lesion_inds_file:
            import json
            with open(args.lesion_inds_file) as f:
                raw = json.load(f)
            # JSON format: [[row_list_or_null, col_list_or_null], ...]
            lesion_inds = [
                (
                    np.array(r, dtype=int) if r is not None else None,
                    np.array(c, dtype=int) if c is not None else None,
                )
                for r, c in raw
            ]
        generate_data(
            model_dir=args.model_dir,
            pattern=args.pattern,
            n_repetitions=args.n_repetitions,
            eval_amp_threshold=args.eval_amp_threshold,
            workers=args.workers,
            backend=args.backend,
            device=args.device,
            overwrite=args.overwrite,
            mode=args.mode,
            n_trials=args.n_trials,
            loads=args.loads,
            delay=args.delay,
            lesion=args.lesion,
            lesion_inds=lesion_inds,
            lesion_scale=args.lesion_scale,
            lesion_tag=args.lesion_tag,
            include_models=args.models,
        )
    elif args.command == "plot-quick-behavior":
        plot_quick_behavior(
            model_dir=args.model_dir,
            pattern=args.pattern,
            mode=args.mode,
            include_models=args.models,
            delay=args.delay,
        )
    elif args.command == "plot-boxplots":
        plot_behavior_boxplots(
            model_dir=args.model_dir,
            pattern=args.pattern,
            mode=args.mode,
            include_models=args.models,
            delay=args.delay,
        )
    elif args.command == "move-low-perf":
        move_low_perf_models(
            model_dir=args.model_dir,
            threshold=args.threshold,
            pattern=args.pattern,
            actually_move=args.actually_move,
        )
    elif args.command == "rename-files":
        rename_old_files_to_balance_stims(args.model_dir, args.pattern)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
