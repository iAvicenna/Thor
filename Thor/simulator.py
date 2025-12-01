#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=bad-indentation, import-error

"""
Created on Thu Jul 17 14:40:31 2025

@author: avicenna
"""

import itertools as it
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
from .models import default_prior_params
from .utils import log2, InternalError

cdir = Path(__file__).parent.resolve()

with open(Path(cdir, "data/data_for_sim"), "rb") as fp:
    data_for_sim = pickle.load(fp)


def generate_simulated_counts(
    npfus,
    nag,
    nrepeats,
    end_dilution,
    sr_names=None,
    seed=None,
    max_seqs=800000,
    min_seqs=400000,
    sample_set=None,
    set_wt=True,
    input_counts=None,
    sd=0.3,
):
    """
    Given npfus, nag (number of antigens) and the end dilution, simulate
    counts in a manner similar to ngs-pfra/rfa exeriments.

    sr_names: if none generates four sera.

    max_seqs, min_seqs: each sample is allocated a total number of sequences
    drawn from uniform with these limits

    sample_set: 0,1 or None, generate_conc_intercepts.

    set_wt: If True, first antigen is set to exact titers, log2_rf and input_proportions
    as the experimental data. Useful when doing relative log2_rf comparisons etc.

    input_counts: If this is supplied, this is used instead of experimental.

    log titer values, log replicative fitness and log input proportions are
    resampled from experimental values with added noise which has standard
    deviation sd.

    concentrations which are the main components of the simulator are generated
    using generate_conc_intercepts
    """

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    max_seqs = 800000
    min_seqs = 400000

    ag_names = [f"Ag{i}" for i in range(nag)]
    if sr_names is None:
        sr_names = ["F09012", "F09016", "F09032", "F95014"]
    x = np.array(range(9))[::-1]

    dil_names = [f"1/{int(end_dilution*2.0**-i)}" for i in x]

    index = generate_index(sr_names, nrepeats, dil_names)

    titer_table, log2_rfs_table, input_props = generate_observables_from_exp(
        ag_names, sr_names, seed, sd, set_wt
    )

    if input_counts is not None:
        input_props = input_counts / input_counts.sum(axis=-1)

    slopes = generate_slopes(sr_names, seed=seed)
    nseqs = generate_nseqs(len(index) - 1, max_seqs, min_seqs, seed=seed)
    neuts = generate_neuts(
        titer_table, slopes, index, max_titer=np.log2(end_dilution / 10)
    )

    conc_intcpts = generate_conc_intercepts(
        neuts, npfus, max_seqs, sample_set, seed=seed
    )

    rfs = 2**log2_rfs_table.values

    real = {
        "slopes": slopes,
        "titer_table": titer_table,
        "log2_rfs": log2_rfs_table,
        "neuts": neuts,
    }

    return (
        simulate(
            input_props,
            index,
            rfs,
            conc_intcpts,
            neuts,
            nseqs,
            ag_names,
            npfus,
            seed=seed,
        ),
        real,
    )


def simulate(
    input_props, index, rfs, intcpts, neuts, nseqs, ag_names, pfus, seed=None
):
    """
    Given the input, simulates ngs-pfra/rfa counts in a manner similar to
    the BB_model in models.py.

    if you want to run this manually all the inputs necessary can be generated
    using the functions inside generate_simulated_counts, see there for details.
    """

    ct_input = 20 + np.log2(
        10000 / pfus
    )  # input ct is arbitrary, exact value does not really matter

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    rng = np.random.default_rng(seed)

    pthr = np.floor(np.log10(1 / np.max(nseqs)))

    ri = intcpts[0, ..., None]
    li = intcpts[1, ..., None]

    def log_conc_fun(x):
        return ri + (li - ri) * np.clip(np.log10(x), pthr, np.inf) / pthr

    fracs = neuts * rfs[None, :] * input_props[None, :]
    sum_fracs = fracs.sum(axis=-1)

    fracs = fracs / sum_fracs[..., None]

    log_concs = np.clip(log_conc_fun(fracs), 1, 18)
    concs = np.exp(log_concs)

    a = (fracs) * concs
    pfu_scales = np.clip(
        rng.normal(0, 0.8, size=sum_fracs.shape[0]), 1 / 20, 20
    )

    dist1 = pm.BetaBinomial.dist(
        n=np.array(nseqs)[:, None], alpha=a, beta=concs - a, size=a.shape
    )

    dist2 = pm.Normal.dist(
        np.log2(sum_fracs).flatten() + pfu_scales,
        default_prior_params["ct_sd"],
    )

    dist3 = pm.Multinomial.dist(n=np.max(nseqs), p=input_props)

    draws1 = pm.draw(dist1, draws=1, random_seed=rng)
    draws2 = pm.draw(dist2, draws=1, random_seed=rng)
    draws3 = pm.draw(dist3, draws=1, random_seed=rng)

    draws1 = np.concatenate([draws3[None, :], draws1])
    draws2 = [ct_input] + list(ct_input - draws2)

    table = pd.DataFrame(
        index=index, columns=["CT"] + list(ag_names), dtype=float
    )

    table.iloc[:, 1:] = draws1
    table.loc[:, "CT"] = draws2

    index = [
        (
            ("INPUT", "", "")
            if x == "INPUT"
            else (
                ("NO SERUM", x.split("_")[1], "")
                if "NO SERUM" in x
                else x.split("_")
            )
        )
        for x in table.index
    ]

    table.index = pd.MultiIndex.from_tuples(
        index, names=["SERUM", "REPEAT", "DILUTION"]
    )

    return table


def generate_index(sera, nrepeats, dilutions):
    """
    this generates the data frame table index given the sera names,
    number of repeats and set of dilutions to be used.
    """

    index = (
        ["INPUT"]
        + [f"NO SERUM_{chr(65+rep)}" for rep in range(nrepeats)]
        + [
            f"{x}_{chr(65+rep)}_{y}"
            for x, rep, y in it.product(sera, range(nrepeats), dilutions)
        ]
    )

    return index


def generate_nseqs(ndata, maxn, minn, seed=None):
    """
    draw total number of sequens for N=ndata number of samples using
    uniform with limits minn, maxn
    """

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    rng = np.random.default_rng(seed)

    return rng.integers(minn, maxn, size=ndata)


def generate_observables_from_exp(
    ag_names, sr_names, seed=0, sd=0.3, set_wt=True
):
    """
    generate titer_table, replicative fitness amd input proprs
    directly from the experiment given antigen and serum names.

    generation is done via resampling from experimental data with replacement
    and adding a noise with standard deviation=sd.

    set_wt: If True, first antigen is set to exact titers, log2_rf and input_proportions
    as the experimental data. Useful when doing relative log2_rf comparisons etc.
    """

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    exp_titer_table = data_for_sim["titers"]
    exp_log2_rfs = data_for_sim["log2_rfs"]
    exp_input_props = np.log2(data_for_sim["input_props"])

    nrows, ncols = exp_titer_table.shape

    rng = np.random.default_rng(seed)
    ag_names = np.array(ag_names)
    sr_names = np.array(sr_names)

    if sr_names.size <= ncols:
        Isr = range(sr_names.size)
    else:
        Isr = list(range(ncols)) + list(
            rng.integers(0, ncols - 1, size=sr_names.size - ncols)
        )

    if ag_names.size <= nrows:
        Iag = range(ag_names.size)
    else:
        Iag = list(range(nrows)) + list(
            rng.integers(0, nrows - 1, size=ag_names.size - nrows)
        )

    titer_table = pd.DataFrame(index=ag_names, columns=sr_names)
    log2_rfs = pd.Series(index=ag_names)

    samples = np.concatenate(
        [
            exp_titer_table.values[np.ix_(Iag, Isr)],
            exp_log2_rfs.values[Iag, None],
            exp_input_props.values[Iag, None],
        ],
        axis=1,
    )

    rv_samples = samples + rng.normal(0, sd, samples.shape)

    titer_table.loc[ag_names, sr_names] = np.round(
        10 * 2 ** rv_samples[:, : sr_names.size]
    ).astype(int)
    log2_rfs.loc[ag_names] = rv_samples[:, sr_names.size]
    input_props = 2 ** rv_samples[:, -1]

    if set_wt:
        titer_table.iloc[0, Isr] = np.round(
            10 * 2 ** exp_titer_table.iloc[0, Isr]
        ).astype(int)
        log2_rfs.iloc[0] = exp_log2_rfs.iloc[0]
        input_props[0] = 2 ** exp_input_props.iloc[0]

    input_props /= input_props.sum()

    return titer_table, log2_rfs, input_props


def generate_slopes(sr_names, seed=None):
    """
    This is the slopes for neutralization curve for each sera.
    """

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    rng = np.random.default_rng(seed)

    sr_names = np.array(sr_names)

    slope_offsets = rng.normal(0.27, 0.05, size=sr_names.size)

    return pd.Series(
        slope_offsets + default_prior_params["s_offset_neut"], index=sr_names
    )


def generate_neuts(titers, slopes, index, max_titer, offset=0):
    """
    given titers, slopes for neut curves, table index and maximum titer
    in log units, generates neutralization curves. dilutions of each
    sample are derived from the index so they should have the correct
    form containing dilutions in form such as _1/5120_. generate_index
    can be used for this purpose. offset can be used to offset dilutions.
    """

    neuts = []
    nag = titers.shape[0]

    def sig_fun(x):
        return 1 - 1 / (1 + np.exp(-x))

    for label in index:
        if label == "INPUT":
            continue
        if "NO SERUM" in label:
            neuts.append(np.ones((nag,)))
            continue

        serum, _, dil = label.split("_")

        x = max_titer - np.log2(float(dil.split("/")[-1]) / 10) + offset
        subtiters = titers.loc[:, serum].values
        logtiters = log2(subtiters)
        drops = max_titer - logtiters
        slope = slopes[serum]

        mu = (x - drops) * slope
        sig = sig_fun(mu)

        if not np.all(sig > 0) or not np.all(~np.isnan(sig)):
            raise InternalError("Simlated neut curve values are lower than zero.")

        neuts.append(sig)

    return np.array(neuts)


def generate_conc_intercepts(
    neuts,
    pfus,
    max_nseqs,
    sample_set=None,
    a=0.53,
    b=-1.09,
    c=1.02,
    mu = 0.75,
    sd1=0.32,
    sd2=0.31,
    seed=0,
):
    """
    default values of a,b,c,mu,sd1,sd2 are fitted based on multiple datasets
    In practice when single datasets are used other parameters are possible.
    2025 dataset parameters are used if sample_set=0 and 2023 dataset
    parameters are used if sample_set=1 (both using the ALL mix). Using
    the generic settings will likely give a conservative result with possibly
    somewhat higher noise for pairwise differences.

    see section Relation Between Noise, PFU and Average Neutralization
    in the paper for how these relations are derived.
    """

    if sample_set is None:
        pass
    elif sample_set == 0:
        a = 0.11
        b = 17.91
        c = -6.55
        mu = 1

        sd1 = 0.41
        sd2 = 0.29
    elif sample_set == 1:
        a = 0.41
        b = 0.01
        c = 0.85
        mu = 0.7

        sd1 = 0.31
        sd2 = 0.3
    else:
        raise ValueError("high_precision can be 0,1 or False")

    if seed is None:
        seed = np.random.SeedSequence().spawn(1)[0]

    rng = np.random.default_rng(seed)

    nvar = neuts.shape[-1]
    prop_threshold = np.floor(np.log10(1 / max_nseqs))

    log_s = np.log(pfus) - 2 * np.log(nvar) + np.log(neuts.sum(axis=-1))

    log_ra_mu = np.exp(a * (log_s + b)) + c
    log_ra_mu = rng.normal(log_ra_mu, sd1)

    s = np.clip(rng.normal(mu, sd2), 0.1, 2)

    log_la_mu = log_ra_mu + np.abs(prop_threshold) * s

    intcpts = np.clip(np.array([log_ra_mu, log_la_mu]), 0, 15)

    return intcpts
