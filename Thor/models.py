#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=bad-indentation, import-error, wrong-import-position, invalid-name, disable=unnecessary-lambda, disable=unsubscriptable-object
# pylint cant see that some pytensor objects are scriptable
"""
Created on Wed Sep 13 17:32:16 2023

@author: avicenna

This module contains two main models: BB_model and BB_mix_model. The latter
is an extended version of the former so better to start reading the first one
to get an understanding. To see example of an input table, see tests folder.
"""

import numbers
import logging
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from .nonparametric import get_pt_ests
from .utils import (
    _convert_dilution,
    _transformation,
    _get_end_dilution,
    BadDilutionFormat,
    BadModelInput,
    BadTableFormat,
    InternalError,
    _catch_log,
)


logger = logging.getLogger(__name__)

_eps = np.finfo(pytensor.config.floatX).eps
__all__ = ["BB_model", "BB_mix_model"]

# default prior params for the full models with fitness and titers
default_prior_params = {
    "ct_sd": 0.1,
    "pfu_sd": 0.4,
    "log2_rf_pop_mean_sd": 0.5,
    "log2_rf_offset_sd": 1,
    "titer_sd": 3,
}

_default_weights_sigmoid_parameters = {
    "sd_weight_lam": 1,
    "scale_weight_sd": 1,
}
_default_neut_sigmoid_parameters = {
    "s_offset_neut": 0.8,
    "s_mu_neut": 0.5,
    "s_sd_neut": 0.25,
}

# this dictionary which contains all parameters for different types of models
# so not all the parameters are used in one given model
_default_concentration_parameters = {
    "conc_intercepts_mu": 7,
    "conc_intercepts_sd": 4,
    "log_conc_mu": 7,
    "log_conc_sd": 4,
    "conc_bounds": [0, 18],
}

_default_parametric_concentration_parameters = {
    "pc_a_mu": 0.5,
    "pc_a_sd": 0.1,
    "pc_b_mu": 1,
    "pc_b_sd": 0.2,
    "pc_c_mu": 1,
    "pc_c_sd": 0.2,
    "pc_s_mu": 0.7,
    "pc_s_sd": 0.3,
    "pc_int_sd": 0.3,
}

default_prior_params.update(_default_concentration_parameters)
default_prior_params.update(_default_parametric_concentration_parameters)
default_prior_params.update(_default_neut_sigmoid_parameters)
default_prior_params.update(_default_weights_sigmoid_parameters)


@_catch_log(logger)
def BB_model(
    table: pd.core.frame.DataFrame,
    input_total_pfus: int,
    prior_params: dict = None,
    subset_variants: List[str] = None,
    concentration_type: str = "linear",
    sd_scale: float = 1,
    xshift: float = 0,
    fixed_input: bool = False,
    ppfu_ratios: List[float] = None,
    sens: bool = False,
) -> Union[pm.model.core.Model, dict]:
    """
    The main model for fitting titers and replicative fitness values to
    a table of ngs-pfra/rfa counts.

    table: this should be pandas table of counts with a multiindex with fields
           SERUM, REPEAT, DILUTION. First column of the table should be
           CT values of each sample (or any other total virus quantifications
           in log units) and the remaining columns variants. The input mixture
           should have its index as (INPUT, nan, nan) if no repeats or of the
           form (INPUT, A, nan) if has repeats A,B etc. Similarly for the NO SERUM
           sample first element of the multiindex is NO SERUM. The serum samples
           should have name of the serum as the first element. See test
           folder for examples of tables. see also _factorize_table

    input_total_pfus: expected number of pfus for the sample. not used in the
                      model itself (unless concentration_type="parametric").
                      mainly used for point estimates of titers.

    prior_params: any parameters supplied here overriders the default_prior_params.

    concentration_type: constant, linear or parametric. smaller samples (such as
                        as 20 variants) can be run with constant which is faster.
                        However linear generally has better cross-validation.
                        use parametric if you want to try to fit titers and rf
                        to a dataset with single repeat per sample however first
                        read the paper to understand what you are doing.

    subset_variants: use the subset of variants given in this input. it adjust the
                     ct values accordingly.

    sd_scale: increase or decrease the sd of any prior which has sd in its
              parametrization using this input. can be used for prior sensitivity
              analysis.

    fixed_input: if False, input proportions are modelled as a Multinomial,
              otherwise they are fixed as the sequencing proportions.

    ppfu_ratios: particle to pfu ratio, if you suspect there is a large deviation
    in ppfu ratio in the input sample, then it maybe good to include this term.
    Otherwise it is 1. It should be a list of floats with length same number of variants.


    xshift: if you want to introduce a shift to your dilution values in the log
            domain. A use case is for instance two experiments where dilution
            range is the same but one has slightly larger reaction volume so
            reaction rate adjustment can be useful.

    sens: if True, some of the priors are changed (such as InverseGamma to Gamma)
          Normal to SkewNormal, which is used for prior sensitivity analysis.

    verbose: if True, _factorize_table prints a summary of the supplied
             table (such as sera, dilutions, antigens found etc)

    returns the model and model_meta which contains information about the model.
    model can be use with pm.sample to fit parameters. Of main interest will be
    the log2_rfs and log2_titers. For more advanced observables like pair-wise
    differences use the sampling functions provided in the samplers.py module
    """

    if not isinstance(input_total_pfus, int):
        warnings.warn(
            f"input_total_pfus must be an integer but is {input_total_pfus}."
            "converting to integer."
        )

        input_total_pfus = int(input_total_pfus)

    model_meta = (
        {}
    )
    # model meta is basically used for storing anything picklable
    # that is model related which may be useful for post-processing
    # but is not stored in an Inference Object obtained at the
    # end of sampling. When possible, it is also used for
    # providing inputs to boiler plate functions at the begginning
    # model_args member can be used to reinitialize a model from
    # scratch when needed for sampling.

    model_meta["name"] = "BB"
    model_meta["model_args"] = {
        "table": table.copy(),
        "input_total_pfus": input_total_pfus,
        "prior_params": prior_params,
        "subset_variants": subset_variants,
        "concentration_type": concentration_type,
        "sd_scale": sd_scale,
        "xshift": xshift,
        "sens": sens,
        "fixed_input": fixed_input,
        "ppfu_ratios": ppfu_ratios,
    }

    if prior_params is None:
        prior_params = {}
    else:
        if not all(x in default_prior_params for x in prior_params):
            warnings.warn(
                "prior_params contain some unknown parameters not"
                "found in default_prior_params. Discarding them."
            )

            prior_params = {
                key: prior_params[key] for key in default_prior_params
            }

    prior_params = dict(default_prior_params, **prior_params)

    table = _preprocess_table(table)

    if subset_variants is None:
        strains = [x for x in table.columns if x != "CT"]
    else:
        strains = subset_variants

    if ppfu_ratios is not None and len(ppfu_ratios) != len(strains):
        raise BadModelInput(
            "length of ppfu ratios should be the same as number of strains"
        )

    table = table.loc[:, ["CT"] + list(strains)]
    model_meta["processed_table"] = table

    _factorize_table(model_meta, table, fixed_input)
    idx = _get_indexers(model_meta["factor_table"])

    obs = _get_obs_for_full_fit(table, strains)
    model_meta["obs"] = obs

    get_pt_ests(model_meta)  # point estimates for some parameters
    pt_ests = model_meta["pt_ests"]
    coords = _get_coords(model_meta, idx)  # naming coords used for pymc models

    # counts of members of various levels like SERUM etc...
    N = _get_counts(table)

    initvals = {
        "log2_rf_offsets": (pt_ests["log2_rf"] - pt_ests["log2_rf_pop_mu"])
        / (2 * pt_ests["log2_rf_pop_sd"]),
        "log2_rf_pop_mean": pt_ests["log2_rf_pop_mu"],
    }

    if concentration_type in ["parametric", "linear"]:
        initvals.update(
            {
                "conc_intercepts": np.zeros(
                    (len(coords["assay_experiment"]), 2)
                )
                + np.array([5, 6])[None, :]
            }
        )
    elif concentration_type == "constant":
        initvals.update(
            {
                "log_conc": np.ones((len(coords["assay_experiment"]),))
                * prior_params["log_conc_mu"]
            }
        )

    model_meta["initvals"] = initvals

    prior_params.update(
        {
            "prop_threshold": np.floor(
                np.min(np.log10((1 / obs["assay"].sum(axis=-1)).astype(float)))
            )
            - 1
        }
    )

    model_meta["updated_prior_params"] = prior_params

    with pm.Model(coords=coords) as model:

        # independent variables and other fixed data
        x = pm.Data(
            "x",
            model_meta["dilution_covariates"]
            + model_meta["model_args"]["xshift"],
        )
        serum_idx = pm.Data("serum_idx", idx["SERUM"])
        experiment_idx = pm.Data("experiment_idx", idx["EXPERIMENT"])
        repeat_idx = pm.Data("repeat_idx", idx["REPEAT"])
        nstrains = len(coords["strain"])
        nsera = len(coords["serum"])
        nassay_samples = len(coords["assay_sample"])
        nassay_experiments = len(coords["assay_experiment"])

        if ppfu_ratios is not None:
            ppfu_ratios = pm.Data("ppfu_ratios", ppfu_ratios)

        input_props = _input_prior(
            nstrains, obs, fixed_input, repeat_idx, ppfu_ratios
        )

        log2_rfs = _log2_rf_prior(
            strains, pt_ests, prior_params, sd_scale, sens
        )

        # titer and slope of neutralization curve and the associated neutralization
        # sigmoid
        if nsera > 0:
            neut = _titers_prior(
                x,
                prior_params,
                nsera,
                nstrains,
                serum_idx,
                pt_ests,
                sd_scale,
                sens,
            )
        else:
            neut = np.array([])

        if not sens:
            pfu_scale = pm.Normal(
                "pfu_scales",
                0,
                prior_params["pfu_sd"],
                size=(nassay_samples,),
                dims=["assay_sample"],
            )
        else:
            pfu_scale = pm.SkewNormal(
                "pfu_scales",
                mu=0,
                sigma=prior_params["pfu_sd"],
                alpha=3,
                size=(nassay_samples,),
                dims=["assay_sample"],
            )

        rfs = 2**log2_rfs

        extended_neut = _extend_neut(neut, N, nstrains)

        log_conc_fun = _concentration_prior(
            nassay_experiments,
            experiment_idx,
            prior_params,
            concentration_type,
            extended_neut,
            input_total_pfus,
            sens,
        )

        # transformed priors
        fracs = extended_neut * rfs * input_props
        sum_fracs = fracs.sum(axis=-1)
        fracs = fracs / sum_fracs[:, None]

        log_concs = log_conc_fun(fracs)
        concs = pm.math.exp(log_concs)
        a = (fracs) * concs

        pm.BetaBinomial(
            "counts",
            n=np.sum(obs["assay"], axis=-1)[:, None],
            alpha=a,
            beta=concs - a,
            observed=obs["assay"],
            size=obs["assay"].shape,
            dims=["assay_sample", "strain"],
        )

        # sum fracs has shape 1 + 1 + nserum_samples
        # first dif_ct is input which is 0 so that one
        # is not used in the likelihood
        pm.Normal(
            "log2_sum_fracs_obs",
            pt.log2(sum_fracs) + pfu_scale,
            prior_params["ct_sd"],
            observed=obs["ct"],
            dims="assay_sample",
        )

        if not fixed_input:

            if ppfu_ratios is not None:
                input_props = input_props * ppfu_ratios[None, :]
                input_props = input_props / input_props.sum()

            pm.Multinomial(
                "input_counts",
                p=input_props,
                n=obs["input"].sum(axis=1),
                observed=obs["input"],
                size=(N["INPUT"],),
            )

    return model, model_meta


@_catch_log(logger)
def BB_mix_model(
    table: pd.core.frame.DataFrame,
    input_total_pfus: int,
    prior_params: dict = None,
    subset_variants: list = None,
    concentration_type: str = "linear",
    sd_scale: float = 1,
    fixed_input: bool = False,
    ppfu_ratios: list = None,
    xshift: float = 0,
) -> pm.model.core.Model:
    """
    inputs and outputs are identical to BB_mix model. The main difference
    is that this is a mixture model which on top of the BB_model adds
    a geometric noise component to the model to model strains with very low
    sequence counts which may not be biologically relevant. Eventhough it
    has somewhat better goodness of fit and cross-validation, it is
    substantially slower and produces very similar log2 titer and rf estimates.
    So mostly for diagnostic and experimental purposes for now.
    """

    if not isinstance(input_total_pfus, int):
        warnings.warn(
            f"input_total_pfus must be an integer but is {input_total_pfus}."
            "converting to integer."
        )

        input_total_pfus = int(input_total_pfus)

    model_meta = {}
    model_meta["model_args"] = {
        "table": table.copy(),
        "input_total_pfus": input_total_pfus,
        "prior_params": prior_params,
        "subset_variants": subset_variants,
        "concentration_type": concentration_type,
        "sd_scale": sd_scale,
        "xshift": xshift,
        "fixed_input": fixed_input,
        "ppfu_ratios": ppfu_ratios,
    }

    model_meta["name"] = "BB_mix"

    if prior_params is None:
        prior_params = {}
    else:
        if not all(x in default_prior_params for x in prior_params):
            warnings.warn(
                "prior_params contain some unknown parameters not"
                "found in default_prior_params. Discarding them."
            )

            prior_params = {
                key: prior_params[key] for key in default_prior_params
            }

    prior_params = dict(default_prior_params, **prior_params)

    table = _preprocess_table(table)

    if subset_variants is None:
        strains = [x for x in table.columns if x != "CT"]
    else:
        strains = subset_variants

    if ppfu_ratios is not None and len(ppfu_ratios) != len(strains):
        raise BadModelInput(
            "length of ppfu ratios should be the same as number of strains"
        )

    table = table.loc[:, ["CT"] + list(strains)]
    model_meta["processed_table"] = table

    _factorize_table(model_meta, table, fixed_input)
    idx = _get_indexers(model_meta["factor_table"])

    obs = _get_obs_for_full_fit(table, strains)
    model_meta["obs"] = obs

    get_pt_ests(model_meta)
    pt_ests = model_meta["pt_ests"]  # point estimates for some parameters
    coords = _get_coords(model_meta, idx)
    N = _get_counts(table)

    initvals = {
        "log2_rf_offsets": (pt_ests["log2_rf"] - pt_ests["log2_rf_pop_mu"])
        / (2 * pt_ests["log2_rf_pop_sd"]),
        "log2_rf_pop_mean": pt_ests["log2_rf_pop_mu"],
    }

    if concentration_type in ["parametric", "linear"]:
        initvals.update(
            {
                "conc_intercepts": np.zeros(
                    (len(coords["assay_experiment"]), 2)
                )
                + np.array([5, 8])[None, :]
            }
        )
    elif concentration_type != "constant":
        initvals.update(
            {
                "conc_intercepts": np.zeros(
                    (len(coords["assay_experiment"]), 2)
                )
                + np.array([5, 6])[None, :]
            }
        )

    initvals.update({"p": 0.5 * np.ones((len(coords["assay_sample"]),))})

    model_meta["initvals"] = initvals

    prior_params.update(
        {
            "prop_threshold": np.floor(
                np.min(np.log10((1 / obs["assay"].sum(axis=-1)).astype(float)))
            )
            - 1
        }
    )

    model_meta["updated_prior_params"] = prior_params

    with pm.Model(coords=coords) as model:

        # independent variables and other fixed data
        x = pm.Data("x", model_meta["dilution_covariates"])

        serum_idx = pm.Data("serum_idx", idx["SERUM"])
        experiment_idx = pm.Data("experiment_idx", idx["EXPERIMENT"])
        repeat_idx = pm.Data("repeat_idx", idx["REPEAT"])

        nstrains = len(coords["strain"])
        nsera = len(coords["serum"])
        nassay_samples = len(coords["assay_sample"])
        nassay_experiments = len(coords["assay_experiment"])

        if ppfu_ratios is not None:
            ppfu_ratios = pm.Data("ppfu_ratios", ppfu_ratios)

        input_props = _input_prior(
            nstrains, obs, fixed_input, repeat_idx, ppfu_ratios
        )

        log2_rfs = _log2_rf_prior(strains, pt_ests, prior_params, sd_scale)

        # titer and slope of neutralization curve and the associated neutralization
        # sigmoid
        neut = _titers_prior(
            x, prior_params, nsera, nstrains, serum_idx, pt_ests, sd_scale
        )

        # probability parameter of the geometric distribution
        # used in mixture
        p = pm.Uniform(
            "p", 0.01, 1 - _eps, size=(nassay_samples,), dims="assay_sample"
        )

        pfu_scale = pm.LogNormal(
            "pfu_scales",
            -prior_params["pfu_sd"] ** 2 / 2,
            prior_params["pfu_sd"],
            size=(nassay_samples,),
            dims=["assay_sample"],
        )

        rfs = 2**log2_rfs

        extended_neut = _extend_neut(neut, N, nstrains)

        log_conc_fun = _concentration_prior(
            nassay_experiments,
            experiment_idx,
            prior_params,
            concentration_type,
            extended_neut,
            input_total_pfus,
        )

        # transformed priors
        fracs = extended_neut * rfs * input_props
        sum_fracs = fracs.sum(axis=-1)
        fracs = fracs / sum_fracs[:, None]

        log_concs = log_conc_fun(fracs)
        concs = pm.math.exp(log_concs)

        a = (fracs * concs).T

        dist1 = pm.BetaBinomial.dist(
            n=np.sum(obs["assay"], axis=-1)[None, :],
            alpha=a,
            beta=concs.T - a,
            size=obs["assay"].T.shape,
        )

        dist2 = pm.Geometric.dist(p, size=obs["assay"].T.shape)

        weights = _weights_prior(input_props, nassay_samples)

        pm.Mixture(
            "counts",
            weights,
            [dist1, dist2],
            observed=obs["assay"].T,
            dims=["strain", "assay_sample"],
        )

        # sum fracs has shape 1 + 1 + nserum_samples first dif_ct is input which
        # is 0 so that one is not used in the likelihood
        pm.Normal(
            "log2_sum_fracs_obs",
            pt.log2(sum_fracs * pfu_scale),
            prior_params["ct_sd"],
            observed=obs["ct"],
            dims="assay_sample",
        )

        if not fixed_input:

            if ppfu_ratios is not None:
                input_props = input_props * ppfu_ratios[None, :]
                input_props = input_props / input_props.sum()

            pm.Multinomial(
                "input_counts",
                p=input_props,
                n=obs["input"].sum(axis=1),
                observed=obs["input"],
                size=(N["INPUT"],),
            )

    return model, model_meta


def _input_prior(nstrains, obs, fixed_input, idx_repeat, ppfu_ratios):

    if not fixed_input:
        input_props = pm.Dirichlet(
            "input_props", a=5 * np.ones((nstrains,)), dims="strain"
        )

    else:
        input_props = obs["input_props"]

        if input_props.shape[0] == 1:
            if input_props.ndim == 2:
                input_props = pm.Data(
                    "input_props", input_props[0, :], dims="strain"
                )[None, :]
            else:
                input_props = pm.Data(
                    "input_props", input_props, dims="strain"
                )[None, :]
            pass
        else:
            input_props = pm.Data("input_props", input_props)[
                idx_repeat - 1, :
            ]

        if ppfu_ratios is not None:
            input_props = input_props / ppfu_ratios[None, :]
            input_props = input_props / input_props.sum()

    return input_props


def _weights_prior(fracs, nassay_samples):

    A = pm.Normal("A", mu=-6, sigma=4, size=(nassay_samples,))
    B = pm.InverseGamma("B", mu=1.5, sigma=0.5, size=(nassay_samples,))

    weights = 1 - pm.math.sigmoid(
        (pm.math.log(fracs) - A[:, None]) * B[:, None]
    )
    weights = pm.math.clip(
        pt.transpose(weights, axes=(1, 0)), -2 * _eps, 1 - 2 * _eps
    )
    weights = pt.transpose(pt.stack([1 - weights, weights]), axes=[1, 2, 0])

    return weights


def _log2_rf_prior(strains, pt_ests, pp, sd_scale=1, sens=False):

    log2_rf_offset = pm.ZeroSumNormal(
        "log2_rf_offsets",
        sigma=sd_scale * pp["log2_rf_offset_sd"],
        shape=(len(strains),),
        dims="strain",
    )

    if not sens:
        log2_rf_pop_mean = pm.Normal(
            "log2_rf_pop_mean",
            pt_ests["log2_rf_pop_mu"],
            sd_scale * pp["log2_rf_pop_mean_sd"],
        )
    else:
        log2_rf_pop_mean = pm.SkewNormal(
            "log2_rf_pop_mean",
            mu=pt_ests["log2_rf_pop_mu"],
            sigma=sd_scale * pp["log2_rf_pop_mean_sd"],
            alpha=3,
        )
    log2_rfs = pm.Deterministic(
        "log2_rfs",
        log2_rf_pop_mean + pt_ests["log2_rf_pop_sd"] * log2_rf_offset,
        dims="strain",
    )

    return log2_rfs


def _titers_prior(
    x, pp, nsera, nstrains, serum_idx, pt_ests, sd_scale=1, sens=False
):

    pop_means = np.array(
        [val["pop_mean"] for val in pt_ests["titers"].values()]
    )
    pop_sds = np.array([val["pop_sd"] for val in pt_ests["titers"].values()])
    mins = np.array([val["bound"][0] for val in pt_ests["titers"].values()])
    maxs = np.array([val["bound"][1] for val in pt_ests["titers"].values()])

    if not sens:
        titers = pm.Normal.dist(
            pop_means[:, None],
            pp["titer_sd"] * sd_scale * pop_sds[:, None],
            size=(nsera, nstrains),
        )

        slopes_offset = pm.InverseGamma.dist(
            mu=pp["s_mu_neut"], sigma=sd_scale * pp["s_sd_neut"], size=nsera
        )
    else:
        titers = pm.SkewNormal(
            "log2_titers",
            mu=pop_means[:, None],
            sigma=3 * sd_scale * pop_sds[:, None],
            alpha=3,
            size=(nsera, nstrains),
            dims=["serum", "strain"],
        )

        slopes_offset = pm.Gamma.dist(
            mu=pp["s_mu_neut"], sigma=sd_scale * pp["s_sd_neut"], size=nsera
        )

    # putting min and max limits in pymc is done via Truncuation
    # similar to the min and max parameters of Stan
    if not sens:
        # no point trying to fit titers too much beyond the min and max dilutions
        # also Truncation not possible in SkewNormal.
        titers = pm.Truncated(
            "log2_titers",
            titers,
            mins[:, None],
            maxs[:, None],
            dims=["serum", "strain"],
        )

    # we do not want negative slopes in titer curves and anything beyond 5
    # is unreasonable for 2 fold dilutions (full drop in neutralization in less
    # than a 2 fold increase in concentration)
    slopes_offset = pm.Truncated(
        "slope_offsets",
        slopes_offset,
        0,
        max(5 - pp["s_offset_neut"], 2),
        dims=["serum"],
    )

    slopes = pp["s_offset_neut"] + slopes_offset

    mu = (x[:, None] - titers[serum_idx, :]) * slopes[serum_idx, None]
    neut = 1 - pm.math.sigmoid(mu)  # sigmoid(x)=1/(1+exp(-x))

    return neut


def _concentration_prior(
    nassay_experiments,
    experiment_idx,
    pp,
    concentration_type,
    neut,
    input_total_pfus,
    sens=False,
):

    if concentration_type == "constant":

        if not sens:
            log_conc = pm.InverseGamma.dist(
                pp["log_conc_mu"], pp["log_conc_sd"], size=nassay_experiments
            )
        else:
            log_conc = pm.Gamma.dist(
                pp["log_conc_mu"], pp["log_conc_sd"], size=nassay_experiments
            )

        # ra, la value means beta concentration is in [exp(ra), exp(la)]
        # and at this point if >15 the BetaBinomial has long become a Binomial
        # if <0 too diffuse. We don't want anything beyond a uniform distribution.
        log_conc = pm.Truncated(
            "log_conc", log_conc, *pp["conc_bounds"], dims="assay_experiment"
        )[experiment_idx]

        return lambda x: pt.tile(log_conc[:, None], (1, x.shape[1]))

    if concentration_type == "linear":
        return _concentration_linear(
            nassay_experiments, experiment_idx, pp, sens
        )

    if concentration_type == "parametric":
        return _concentration_parametric(
            neut, input_total_pfus, experiment_idx, pp
        )

    logger.error(
        "concentration_type can only be constant, linear or "
        f"parametric but was {concentration_type}. Setting it to linear."
    )


def _concentration_parametric(neut, pfus, experiment_idx, pp):
    """
    c_y0, c_rate, la_s are parameters that are determined using
    multiple datasets and studying the relation between pfus, neut,
    number of variants and concentration parameters. See the paper
    section Relation Between Noise, PFU and Average Neutralization
    for more details.
    """

    nstrains = neut.shape[1]
    sum_neut = neut.sum(axis=-1)

    a = pm.Normal("pc_a", pp["pc_a_mu"], pp["pc_a_sd"])
    b = pm.Normal("pc_b", pp["pc_b_mu"], pp["pc_b_sd"])
    c = pm.Normal("pc_c", pp["pc_c_mu"], pp["pc_c_sd"])
    s = pm.Normal("pc_s", pp["pc_s_mu"], pp["pc_s_sd"])

    log_s = np.log(pfus) - 2 * pm.math.log(nstrains) + pm.math.log(sum_neut)
    I = [
        list(experiment_idx.eval()).index(i)
        for i in sorted(list(set(experiment_idx.eval())))
    ]
    log_s = log_s[I]

    mu1 = pm.math.exp(a * (log_s - b)) + c
    mu2 = mu1 + s * abs(pp["prop_threshold"])

    mu = pt.concatenate([mu1[:, None], mu2[:, None]], axis=1)

    intcpts = pm.Normal.dist(mu=mu, sigma=pp["pc_int_sd"])
    intcpts = pm.Truncated(
        "conc_intercepts",
        intcpts,
        *pp["conc_bounds"],
        transform=pm.distributions.transforms.ordered,
    )

    ri = intcpts[experiment_idx, 0][:, None]
    li = intcpts[experiment_idx, 1][:, None]

    return (
        lambda x: ri
        + (li - ri)
        * pm.math.clip(pt.math.log10(x), pp["prop_threshold"], np.inf)
        / pp["prop_threshold"]
    )


def _concentration_linear(nassay_experiments, experiment_idx, pp, sens=False):
    if not sens:
        intcpts = pm.InverseGamma.dist(
            mu=pp["conc_intercepts_mu"],
            sigma=pp["conc_intercepts_sd"],
            size=(nassay_experiments, 2),
        )
    else:
        intcpts = pm.Gamma.dist(
            mu=pp["conc_intercepts_mu"],
            sigma=pp["conc_intercepts_sd"],
            size=(nassay_experiments, 2),
        )

    # intcps ra,la means beta concentration is in [exp(ra), exp(la)]
    # and at this point if >15 the BetaBinomial has long become a Binomial
    # if <0 too diffuse. We don't want anything too beyond a uniform distribution.
    intcpts = pm.Truncated(
        "conc_intercepts",
        intcpts,
        *pp["conc_bounds"],
        size=(nassay_experiments, 2),
        transform=pm.distributions.transforms.ordered,
    )

    ri = intcpts[:, 0][experiment_idx][:, None]
    li = intcpts[:, 1][experiment_idx][:, None]

    return (
        lambda x: ri
        + (li - ri)
        * pm.math.clip(pt.math.log10(x), pp["prop_threshold"], np.inf)
        / pp["prop_threshold"]
    )


def _get_counts(table):
    return {
        "INPUT": len(
            [x for x in table.index.get_level_values("SERUM") if x == "INPUT"]
        ),
        "NO SERUM": len(
            [
                x
                for x in table.index.get_level_values("SERUM")
                if x == "NO SERUM"
            ]
        ),
        "SERUM": len(
            [
                x
                for x in table.index.get_level_values("SERUM")
                if x not in ["INPUT", "NO SERUM"]
            ]
        ),
    }


def _get_coords(model_meta, idx):

    level_sets = model_meta["level_sets"]
    sera = [x for x in level_sets["SERUM"] if x not in ["NO SERUM", "INPUT"]]

    if sorted(list(set(idx["SERUM"]))) != list(range(len(set(idx["SERUM"])))):
        raise InternalError('idx["SERUM"] skips certain integers.')

    if len(set(idx["SERUM"])) != len(level_sets["SERUM"]):
        raise InternalError(
            'idx["SERUM"] length different from level_sets["SERUM"] length.'
        )

    coords = {
        "repeat": [x for x in level_sets["REPEAT"] if x != ""],
        "dilution": [x for x in level_sets["DILUTION"] if x != ""],
        "sample": np.array(list(map(_join, level_sets["SAMPLE"])))[
            idx["SAMPLE"]
        ],
        "strain": model_meta["strains"],
        "_strain": model_meta[
            "strains"
        ],  #  used for pairwise difference observable tracking
        "assay_experiment": np.array(
            list(map(_join, level_sets["ASSAY_EXPERIMENT"]))
        ),
        "serum": np.array(sera),
    }

    coords["assay_sample"] = np.array(
        [x for x in coords["sample"] if "INPUT" not in x]
    )

    coords["serum_assay_sample"] = np.array(
        [
            x
            for x in coords["sample"]
            if "INPUT" not in x and "NO SERUM" not in x
        ]
    )

    return coords


def _get_obs_for_full_fit(table, strains):

    I = {}
    I["INPUT"] = [x for x in table.index if "INPUT" in x]
    I["ASSAY"] = [x for x in table.index if "INPUT" not in x]

    input_props_obs = (
        table.loc[I["INPUT"], strains]
        / table.loc[I["INPUT"], strains].values.sum(axis=-1)[:, None]
    ).values
    assay_obs = table.loc[I["ASSAY"], strains].values
    input_obs = table.loc[I["INPUT"], strains].values

    ct_obs = (
        table.loc[I["INPUT"], "CT"].values.mean()
        - table.loc[I["ASSAY"], "CT"].values
    )

    obs = {
        "ct": ct_obs,
        "input": input_obs,
        "input_props": input_props_obs,
        "assay": assay_obs,
    }

    return obs


def _extend_neut(neut, N, nstrains):
    """
    This extends tensor neut that (normally tracks n neutralization amounts
    in serum samples) to account for no serum samples.

    It is extended at the start by rows of 1 since there is no
    neutralization in no serum. it can also extend p which is a
    tensor used in the experimental BB_mix model.
    """

    if N["NO SERUM"] > 0:
        neut = pt.concatenate(
            [np.ones((N["NO SERUM"], nstrains)), neut], axis=0
        )

    return neut


def _preprocess_table(
    table: pd.core.frame.DataFrame,
) -> pd.core.frame.DataFrame:
    """
    sort the table so that INPUT is first, NO SERUM is second,
    SERA are third. SERA are ordered according to log dilution values and
    then everything else according to REPEAT level

    Some basic checks are also carried out. ordered table is produced.
    """
    table = table.copy()

    if isinstance(table.index, pd.MultiIndex):
        table.index = pd.MultiIndex.from_frame(
            table.index.to_frame().fillna("")
        )
    else:
        table.index = table.index.fillna("")

    if len(set(table.index)) != len(table.index):
        raise BadTableFormat("Table index should be unique.")

    if not all(
        x in table.index.names for x in ["SERUM", "REPEAT", "DILUTION"]
    ):
        raise BadTableFormat(
            "Table index should contain SERUM, REPEAT, DILUTION."
        )
    if not table.columns[0] == "CT":
        raise BadTableFormat("Table's first column should be CT.")
    if not "INPUT" in table.index.get_level_values("SERUM"):
        raise BadTableFormat(
            "Table index should contain atleast one label with SERUM level INPUT."
        )

    table = table.reorder_levels(["SERUM", "REPEAT", "DILUTION"])

    I = sorted(table.index.tolist(), key=_sort_table_index)

    table = table.loc[I, :]

    sera = table.index.get_level_values("SERUM").tolist()

    if not table.shape[0] > sera.count("INPUT"):
        raise BadTableFormat(
            "Table must contain atleast one index which is not "
            "INPUT for SERUM level."
        )

    if not (
        ("INPUT" in sera and sera.index("INPUT") == 0)
        or len(sera) - sera[::-1].index("INPUT") == sera.count("INPUT")
    ):
        raise InternalError(
            "Either INPUT does not have zero index or "
            "start of non INPUT sera != # INPUT repeats."
        )

    if "NO SERUM" in sera and (
        sera.index("NO SERUM") != sera.count("INPUT")
        or len(sera) - sera[::-1].index("NO SERUM") - sera.index("NO SERUM")
        != sera.count("NO SERUM")
    ):
        raise InternalError(
            "Either start of NO SERUM != # INPUT repeats "
            "or start of non NO SERUM sera != # INPUT + "
            "NO SERUM repeats."
        )

    if not all(isinstance(x, numbers.Number) for x in table.values.flatten()):
        logger.error("Table contains non-numeric values. Trying to convert.")
        try:
            table = table.astype(float)
        except ValueError:
            raise BadTableFormat(
                "Table contains non-convertible non-numeric values."
            )

    for label in table.index:
        if label[0] not in ["INPUT", "NO SERUM"]:
            # check that dilutions are log convertable or else is nan
            dilution = _convert_dilution(label[2], np.log, 5120)
            if np.isnan(dilution) or str(label[2])[:2] != r"1/":
                raise BadDilutionFormat(
                    f"Serum entry {label} has invalid dilution. "
                    f"It should be of the form 1/X"
                )

    return table


def _factorize_table(
    model_meta: dict,
    table: pd.core.frame.DataFrame,
    fixed_input: bool,
) -> None:
    """
    Given a table with multiindex containing Serum, Repeat, Dilution info,
    it produces a factor_table where these are columns. It also creates:

    1- some extra columns which are combinations of these (for book keeping).
    2- a seperate level_sets dict, which gives (in order)
    what are the non-numerical values for each columns factors are.
    3- it also produces dilution covariates which are numerical values
    associated to each dilution level (in default parameters, lowest dilution
    is associated highest value so a list of dilutions like 1/5120, ..., 1/20
    would be assinged values 0,...,8).

    All are be stored in the model_meta dict.

    This function assumes that the table is ordered so that INPUT comes first
    NO SERUM comes second and SERA come following. Call _preprocess_table before
    this function to get a table ordered this way.
    """
    strains = table.columns[1:].to_numpy()
    level_sets = {}

    factor_table = pd.DataFrame(
        columns=["SAMPLE", "EXPERIMENT", "SERUM", "REPEAT", "DILUTION"]
        + list(table.columns),
        index=range(table.shape[0]),
    )

    factor_table.loc[:, table.reset_index().columns] = table.reset_index()

    factor_table.loc[:, "EXPERIMENT"] = list(
        zip(factor_table.loc[:, "SERUM"], factor_table.loc[:, "DILUTION"])
    )

    factor_table.loc[:, "SAMPLE"] = list(
        zip(
            factor_table.loc[:, "SERUM"],
            factor_table.loc[:, "DILUTION"],
            factor_table.loc[:, "REPEAT"],
        )
    )
    for key in ["SERUM", "REPEAT", "DILUTION", "EXPERIMENT", "SAMPLE"]:
        levels, level_set = pd.factorize(factor_table[key])
        if key == "EXPERIMENT" and any(x[0] == "INPUT" for x in level_set):
            level_set = [x for x in level_set if x[0] != "INPUT"]
            levels = [int(x - 1) if x != 0 else np.nan for x in levels]
        elif key == "DILUTION" and any(x == "" for x in level_set):
            level_set = [x for x in level_set if x != ""]
            levels = [int(x - 1) if x != 0 else np.nan for x in levels]
        elif key == "SERUM" and any(
            x in ["INPUT", "NO SERUM"] for x in level_set
        ):
            s0 = int("INPUT" in level_set) + int("NO SERUM" in level_set)
            level_set = [
                x for x in level_set if x not in ["INPUT", "NO SERUM"]
            ]
            levels = [int(x - s0) if x >= s0 else np.nan for x in levels]
        else:
            level_set = level_set.values.tolist()

        factor_table[key] = levels
        level_sets[key] = level_set

    if factor_table.loc[:, "SAMPLE"].values.tolist() != list(
        range(table.shape[0])
    ):
        raise InternalError("SAMPLE in factor table misses some integers.")

    level_sets["SERUM_EXPERIMENT"] = [
        x
        for x in level_sets["EXPERIMENT"]
        if x[0] not in ["NO SERUM", "INPUT"]
    ]

    level_sets["ASSAY_EXPERIMENT"] = [
        x for x in level_sets["EXPERIMENT"] if x[0] not in ["INPUT"]
    ]

    level_sets["ASSAY_SAMPLE"] = [
        x for x in level_sets["SAMPLE"] if x[0] != "INPUT"
    ]

    level_sets["SERUM_SAMPLE"] = [
        x
        for x in level_sets["SAMPLE"]
        if x[0] != "INPUT" and x[0] != "NO SERUM"
    ]

    model_meta["end_dilution"] = _get_end_dilution(level_sets["DILUTION"])

    dilution_covariates = np.array(
        [
            _convert_dilution(
                level_sets["DILUTION"][int(x)],
                _transformation,
                1 / model_meta["end_dilution"],
            )
            for x in factor_table.loc[:, "DILUTION"].values
            if not np.isnan(x)
        ]
    )

    model_meta["factor_table"] = factor_table
    model_meta["level_sets"] = level_sets
    model_meta["strains"] = strains
    model_meta["dilution_covariates"] = dilution_covariates

    if fixed_input:
        assay_repeats = np.array(
            [
                x
                for x, y in zip(
                    factor_table.loc[:, "REPEAT"].values.astype(int),
                    factor_table.loc[:, "EXPERIMENT"].values,
                )
                if not isinstance(y, float) or not np.isnan(y)
            ]
        )

        input_repeats = np.array(
            [
                x
                for x, y in zip(
                    factor_table.loc[:, "REPEAT"].values.astype(int),
                    factor_table.loc[:, "EXPERIMENT"].values,
                )
                if isinstance(y, float) and np.isnan(y)
            ]
        )

        if not (
            set(input_repeats) == set(assay_repeats)
            or len(set(input_repeats)) == 1
        ):
            raise BadTableFormat(
                "If fixed_input is True, repeat levels for INPUT must "
                "be the same as others or input must have a single repeat."
            )

    logger.info("Found level sets:")

    for key in ["SERUM", "DILUTION", "REPEAT"]:
        if len(level_sets[key]) == 0:
            logger.info("WARNING: %s not found in table", key)
        else:
            logger.info("%s: %s", key, level_sets[key])

    logger.info("Found strains:\n%s", strains.tolist())


def _sort_levels(level):
    return [level != "INPUT", level != "NO SERUM"]


def _get_indexers(factor_table):
    """
    Given a factor_table where each column such as SERUM,SAMPLE (level names) contains
    level values, this returns an idx which is a dictionary that maps these level
    names to lists where each ith entry of the list is the corresponding
    level index for the ith row of the table.
    """
    idx = {}
    idx["EXPERIMENT"] = [
        int(x)
        for x in factor_table.loc[:, "EXPERIMENT"].values
        if not isinstance(x, float) or not np.isnan(x)
    ]

    idx["SERUM"] = np.array(
        [
            x
            for x in factor_table.loc[:, "SERUM"].values
            if not isinstance(x, float) or not np.isnan(x)
        ]
    ).astype(int)

    idx["REPEAT"] = np.array(
        [
            x
            for x, y in zip(
                factor_table.loc[:, "REPEAT"].values.astype(int),
                factor_table.loc[:, "EXPERIMENT"].values,
            )
            if not isinstance(y, float) or not np.isnan(y)
        ]
    )

    idx["SAMPLE"] = factor_table.loc[:, "SAMPLE"].values.astype(int)

    return idx


def _sort_sera(level):
    return [level != "INPUT", level != "NO SERUM", level]


def _sort_table_index(x):
    """
    for dilution sorting purposes one does not need to specify a starting
    dilution so a default value of 5120 is used but anything compatible with
    a log transformation is fine
    """
    return _sort_sera(x[0]) + [x[1]] + [_convert_dilution(x[2], np.log, 5120)]


def _join(elems, sep="_"):
    return f"{sep}".join([elem for elem in elems if elem != ""])
