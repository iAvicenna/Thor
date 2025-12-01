#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=bad-indentation, import-error, wrong-import-position
"""
Created on Wed Jul 10 16:05:09 2024

@author: avicenna
"""

import warnings
from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import mode
from .utils import (
    _convert_dilution,
    _transformation,
    InternalError,
    UnacceptableInput,
    BadTableFormat,
)

__ALL__ = ["compute_log_rf_pt_est", "neut_pt_est"]


def get_pt_ests(model_meta: dict) -> None:
    """
    using the model_meta initialized from a model in the models.py module,
    compute point estimates for replicative fitness and titers. see
    relevant functions below to see how.
    """

    pt_ests = {}

    pt_ests["log2_rf"] = compute_log_rf_pt_est(model_meta)
    pt_ests["log2_rf_pop_sd"] = pt_ests["log2_rf"].std()
    pt_ests["log2_rf_pop_mu"] = pt_ests["log2_rf"].mean()

    if (
        "level_sets" in model_meta
        and len(
            set(model_meta["level_sets"]["SERUM"]).difference(
                ["INPUT", "NO SERUM"]
            )
        )
        > 0
    ):

        pfu_table = pfus_from_counts(model_meta, pt_ests["log2_rf"])
        pt_ests["titers"] = spearman_karber(pfu_table, model_meta)

        model_meta["pfu_table"] = pfu_table

    model_meta["pt_ests"] = pt_ests


def spearman_karber(
    pfu_table: pd.core.frame.DataFrame, model_meta: dict
) -> dict:
    """
    This is the spearman_karber non-parametric estimator for titers
    (doi.org/10.2307/2335002).

    Ayer,Brunk,Ewing,Reid,Silverman algorithm is used to construct
    monotonically decrease neutralizations from noisy data
    (DOI: 10.1214/aoms/1177728423).

    Note that SK estimated titers are only used to build a weakly informative
    prior by calculating for each serum where the estimated population mean titer
    is and what is the sd of population titer distribution.

    Individual fitted titers are not used in any downstream analysis, although a
    good sanity check could be to compare Bayesian fitted titers to SK titers
    to see they roughly match in their means and sd.

    pfu_table can be obtained via pfus_from_counts.
    """

    strains = model_meta["strains"]
    level_sets = model_meta["level_sets"]

    serum_indices = [
        ind
        for ind, s in enumerate(level_sets["SERUM"])
        if s not in ["INPUT", "NO SERUM"]
    ]

    noserum_indices = [
        ind
        for ind, s in enumerate(level_sets["EXPERIMENT"])
        if s == ("NO SERUM", "")
    ]

    noserum_pfus = pfu_table[pfu_table.EXPERIMENT.isin(noserum_indices)]
    repeats = sorted(list(set(noserum_pfus.loc[:, "REPEAT"].values)))
    noserum_pfus = np.array(
        [
            noserum_pfus[noserum_pfus.REPEAT.isin([i])].loc[:, strains]
            for i in repeats
        ]
    ).astype(float)
    noserum_pfus[noserum_pfus == 0] = np.nan

    serum_to_titer_est = {}

    for inds in serum_indices:

        serum_pfus = pfu_table[pfu_table.SERUM.isin([inds])]

        repeats = sorted(list(set(serum_pfus.loc[:, "REPEAT"].values)))

        repeat_pfus = np.array(
            [
                serum_pfus[serum_pfus.REPEAT.isin([i])].loc[:, strains]
                for i in repeats
            ]
        ).astype(float)

        dilution_covariates, xvals, step = _get_dilution_covariates(
            serum_pfus, repeats, level_sets, model_meta
        )

        dilution_covariates += model_meta["model_args"]["xshift"]

        p = np.clip(repeat_pfus / noserum_pfus, 0, 1)
        p = np.apply_along_axis(
            ayer_brunk_ewing_reid_silverman, axis=1, arr=p[:, ::-1, :]
        )[:, ::-1, :]

        left = np.zeros((p.shape[0], 1, p.shape[2]))
        right = np.ones((p.shape[0], 1, p.shape[2]))
        p_ext = np.concatenate([left, p, right], axis=1)
        titers = np.nansum(
            (p_ext[:, 1:, :] - p_ext[:, :-1, :]) * xvals[None, :, None], axis=1
        )

        titers = _interpolate(titers, xvals, dilution_covariates)

        with warnings.catch_warnings():
            # for p which are both nan coming from viruses where both noserum
            # repeats are 0
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            var = step**2 * np.sum(p * (1 - p), axis=1) / noserum_pfus[:, 0, :]
            sd = np.sqrt(0.5 * np.nanmean(var, axis=0))

        serum_to_titer_est[inds] = {
            "gmt": np.nanmean(titers, axis=0),
            "sd": sd,
            "pop_mean": np.nanmean(titers),
            "pop_sd": np.nanstd(titers),
            "bound": [
                np.min(dilution_covariates) - step,
                np.max(dilution_covariates) + step,
            ],
        }

    return serum_to_titer_est


def ayer_brunk_ewing_reid_silverman(
    p: np.ndarray, nparents: Union[np.ndarray, None] = None
) -> np.ndarray:
    """
    This is an algorithm which turns non monotonic sequences
    into those which are monotonically decreasing.
    (DOI: 10.1214/aoms/1177728423)

    p should be a sequence of proportions. you don't need to initialize
    nparents, it is a part of the recursion.
    """
    if nparents is None:
        nparents = np.ones((p.size,)).astype(int)

    if np.all(np.isnan(p)):
        return p

    dif = np.diff(p)

    if np.all(dif <= 0):
        return [p[i] for i in range(len(p)) for _ in range(nparents[i])]

    I = np.argwhere(dif > 0).flatten()[0]
    new_p = np.array(
        list(p[:I]) + [np.nansum(p[I : I + 2]) / 2] + list(p[I + 2 :])
    )
    new_nparents = np.array(
        list(nparents[:I])
        + [nparents[I : I + 2].sum()]
        + list(nparents[I + 2 :])
    )

    return ayer_brunk_ewing_reid_silverman(new_p, new_nparents)


def pfus_from_counts(
    model_meta: dict, log2_rf: np.ndarray
) -> pd.core.frame.DataFrame:
    """
    using the model_meta initialized from a model in the models.py module,
    and log2_rf point estimates, compute the point estimate for number of
    pfus left for each variant after neutralization.
    """

    strains = model_meta["processed_table"].columns[1:]
    factor_table = model_meta["factor_table"]
    level_sets = model_meta["level_sets"]
    input_total_pfus = model_meta["model_args"]["input_total_pfus"]

    if log2_rf.size != strains.size:
        raise UnacceptableInput(
            'log2_rf should have the same size as model_meta["strains"].'
        )
    if "CT" not in factor_table.columns:
        raise InternalError("factor_table should have contained CT.")
    if level_sets["EXPERIMENT"][0][0] != "NO SERUM":
        raise InternalError(
            "First element of experiment level sets should have been "
            "tuple starting with NO SERUM."
        )

    rf = 2**log2_rf

    I = [
        label
        for ind, (label, row) in enumerate(factor_table.iterrows())
        if ~np.isnan(row["EXPERIMENT"])
    ]

    pfus = factor_table.loc[I, :].copy()

    i0 = [
        label
        for ind, (label, row) in enumerate(factor_table.iterrows())
        if row["EXPERIMENT"]
        == level_sets["EXPERIMENT"].index(("NO SERUM", ""))
    ]

    if len(i0) == 0:
        raise InternalError(
            "factor_table contains no rows whose EXPERIMENT "
            "value equals NO SERUM's EXPERIMENT index."
        )

    ct0 = factor_table.loc[i0, :]

    for label, row in pfus.iterrows():

        row_counts = row[strains].values
        row_inv_counts = row_counts / rf
        row_inv_props = row_inv_counts / row_inv_counts.sum()
        repeat = row["REPEAT"]

        ct_noserum = ct0[ct0.REPEAT.isin([repeat])].loc[:, "CT"].values[0]

        scale = 2 ** (ct_noserum - row["CT"])

        pfus.loc[label, strains] = (
            input_total_pfus * row_inv_props * scale
        ).astype(int)

    return pfus


def compute_log_rf_pt_est(model_meta: dict) -> np.ndarray:
    """
    using the model_meta initialized from a model in the models.py module,
    compute point estimates for log2 replicative fitness.
    """

    I = {}
    table = model_meta["processed_table"]

    I["INPUT"] = [x for ind, x in enumerate(table.index) if x[0] == "INPUT"]
    I["NO SERUM"] = [
        x for ind, x in enumerate(table.index) if x[0] == "NO SERUM"
    ]

    ppfus_ratios = model_meta["model_args"].get("ppfu_ratios", None)

    cols = list(table.columns)
    i0 = cols.index("CT") + 1
    nstrains = len(cols[i0:])

    input_counts = table.loc[I["INPUT"], cols[i0:]].values.astype(float)
    noserum_counts = table.loc[I["NO SERUM"], cols[i0:]].values.astype(float)

    input_cts = table.loc[I["INPUT"], "CT"].values.astype(float)
    noserum_cts = table.loc[I["NO SERUM"], "CT"].values.astype(float)

    input_fracs = input_counts / input_counts.sum(axis=-1)[:, None]

    if ppfus_ratios is not None:
        input_fracs = input_fracs / np.array(ppfus_ratios)[None, :]
        input_fracs = input_fracs / input_fracs.sum()

    noserum_fracs = noserum_counts / noserum_counts.sum(axis=-1)[:, None]

    reg_input = np.tile(
        1 / (2 * input_counts.sum(axis=-1))[:, None], (1, nstrains)
    )
    reg_noserum = np.tile(
        1 / (2 * noserum_counts.sum(axis=-1))[:, None], (1, nstrains)
    )

    if (input_fracs[input_fracs == 0]).size > 0:
        input_fracs[input_fracs == 0] = reg_input[input_fracs == 0]

    if (noserum_fracs == 0).size > 0:
        noserum_fracs[noserum_fracs == 0] = reg_noserum[noserum_fracs == 0]

    if input_counts.shape[0] != noserum_counts.shape[0]:
        input_fracs = input_fracs.mean(axis=0)[None, :]
        input_cts = input_cts.mean()

        if input_counts.shape[0] == 1:
            input_cts = np.array([input_cts])

    rf_pt_est = noserum_fracs / input_fracs
    rf_pt_est *= (np.array(2**-noserum_cts) / 2**-input_cts)[:, None]
    log2_rf_pt_est = np.log2(rf_pt_est.astype(float))

    if log2_rf_pt_est.ndim == 2:
        log2_rf_pt_est = log2_rf_pt_est.mean(axis=0)
    elif log2_rf_pt_est.ndim > 2:
        raise InternalError(
            "log2_rf_pt_est should have been 2 dimensional but "
            f"it has shape {log2_rf_pt_est.shape}."
        )

    return log2_rf_pt_est


def _interpolate(titers, xvals, dilution_covariates):

    return np.interp(titers, xvals, dilution_covariates)


def _get_dilution_covariates(serum_pfus, repeats, level_sets, model_meta):
    # note that this works on a table called serum_pfus which is the
    # pfus for a given serum across different dilutions. in particular
    # it only contains repeats for just one serum and the covariates it returns
    # are for that serum.

    dilution_covariates = np.array(
        [
            [
                _convert_dilution(
                    level_sets["DILUTION"][int(x)],
                    _transformation,
                    1 / model_meta["end_dilution"],
                )
                for x in serum_pfus.loc[
                    serum_pfus.REPEAT.isin([i]), "DILUTION"
                ].values
                if not np.isnan(x)
            ]
            for i in repeats
        ]
    )

    if not all(
        np.all(row1 == row2)
        for row1 in dilution_covariates
        for row2 in dilution_covariates
    ):
        si = int(serum_pfus["SERUM"].values[0])
        raise BadTableFormat(
            "You can not have different dilutions between SERUM repeats. "
            f'This error occured in serum {level_sets["SERUM"][si]}.'
        )

    dilution_covariates = dilution_covariates[0, :]

    steps = -np.diff(dilution_covariates)
    if len(set(steps)) > 1:
        si = int(serum_pfus["SERUM"].values[0])
        warnings.warn(
            "Multiple steps between dilutions found for serum "
            f'{level_sets["SERUM"][si]}. Non-parameteric estimates '
            "may be unreliable. Increasing titer_sd in the prior_params"
            "might be useful."
        )

    step = mode(steps).mode

    steps = [0] + list(steps)
    xvals = np.array(
        [-step / 2 + np.sum(steps[: i + 1]) for i in range(len(steps))]
    )
    xvals = np.append(xvals, [xvals[-1] + step])

    dilution_covariates = dilution_covariates + step / 2
    dilution_covariates = np.append(
        dilution_covariates, [dilution_covariates[-1] - step]
    )

    return dilution_covariates, xvals, step
