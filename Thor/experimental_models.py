#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:23:49 2026

@author: avicenna
"""

import logging
import warnings
from typing import Union, List

import pandas as pd
import pymc as pm
import numpy as np
import pytensor.tensor as pt
import pytensor

from .models import default_prior_params, _get_indexers, _factorize_table,\
  _get_coords, _get_counts, _get_obs_for_full_fit, _preprocess_table,\
    get_pt_ests, _input_prior, _extend_neut, _log2_rf_prior,\
      _concentration_prior, _titers_prior
      
from .utils import _catch_log, BadModelInput

__all__ = ["BB_ct_eiv_model", "BB_mix_model"]
logger = logging.getLogger(__name__)

_extra_prior_params = {
    "gamma_mu": 0.25,
    "gamma_sd": 0.10,
}


_default_weights_sigmoid_parameters = {
    "sd_weight_lam": 1,
    "scale_weight_sd": 1,
}

experimental_prior_params = dict(default_prior_params, **_extra_prior_params)
experimental_prior_params = dict(experimental_prior_params, **_default_weights_sigmoid_parameters)

_founder_prior_params = {
    "ct_slope_excess": 0.20,
    "ct_slope_excess_sd": 0.75,
    "founder_k_mu": 3.3,
    "founder_k_sd": 1.0,
}

founder_prior_params = dict(default_prior_params, **_founder_prior_params)
_eps = np.finfo(pytensor.config.floatX).eps



@_catch_log(logger)
def BB_ct_eiv_model(
    table: pd.core.frame.DataFrame,
    input_total_pfus: int,
    prior_params: dict = None,
    subset_variants: List[str] = None,
    concentration_type: str = "linear",
    use_xlatent: bool = False,
    sd_scale: float = 1,
    xshift: float = 0,
    fixed_input: bool = False,
    ppfu_ratios: List[float] = None,
    sens: bool = False,
) -> Union[pm.model.core.Model, dict]:
    """
    models.BB_model with two changes, both confined to the ct likelihood.

    models.BB_model predicts dct with log2(sum_fracs) directly, i.e. it assumes
    a one to one relation between dct and log2 total virus. Across five
    datasets that relation is straight but its slope is not 1: fitted values
    run 1.02 to 1.33, and forcing 1 leaves 27% of wells outside their own 2
    sigma and costs 80 to 250 ELPD. So

        dct ~ Normal(ref + b*(log2 sum_fracs - ref), ct_sd)

    with b >= 1 shared across the whole dataset, and ref the mean predicted
    log2 sum_fracs of the NO SERUM wells. b = 1 recovers models.BB_model.

    Anchoring on NO SERUM matters: those wells have neut == 1 by construction,
    so the correction is exactly zero there and b describes only the serum
    induced part of the drop. Without the anchor the tilt would have to be
    absorbed by log2_rf_pop_mean, which the counts cannot see (it cancels in
    the normalised fracs) and which carries a Normal(., 0.5) prior - a shift of
    up to 3.9 prior sd across these datasets.

    b is one number for the dataset, not one per serum. Per-serum slopes are
    supported by the data in the large panels but make the posterior funnel,
    especially with 2-3 sera.

    b >= 1 is imposed as b = 1 + exp(normal): the correction can steepen dct,
    never flatten it. A freely signed version drifted to 0.77 on one serum.

    2. The ct WIDTH is founder limited. A well's total is a sum over its S
       surviving founders, whose individual yields are very unequal, so

           sd(dct) = sqrt(ct_sd^2 + founder_k^2 / S)
           S       = input_total_pfus * sum_v neut_v p_v

       rather than a single ct_sd. The k^2/S law is measured directly in the
       NO SERUM wells - no serum, no ct, no fitted model - where it holds over a
       300x range of founder number. This matters most in small panels, where S
       reaches single digits; in 134-141 variant panels S rarely drops below
       ~180 and the term is second order. Note it makes input_total_pfus load
       bearing, where models.BB_model documents it as unused.

       Caveat: widening sigma at low S also flattens the likelihood there, so
       the fit chases those wells less hard. On these datasets the mean
       |residual| of the most neutralised third rose from 0.77 to 0.90 when this
       term was added alone.

       The counts likelihood is deliberately NOT given the same treatment: it is
       the correct mechanism for the BetaBinomial overdispersion too, but doing
       so made the posterior multimodal on the 134 variant panel (max rhat 2.37,
       two variants with one chain in a different mode) because a variant's
       titer would then set both its expected proportion and its own dispersion.

    Extra prior parameters on top of models.default_prior_params:
      ct_slope_excess, ct_slope_excess_sd: lognormal prior on b - 1, so the
        default is a median slope of 1.2 spanning roughly 1.05 to 1.9.
      founder_k_mu, founder_k_sd: prior on k. Only the ct channel informs it,
        so it is appreciably prior driven; measure k in the NO SERUM wells.

    Everything below is models.BB_model unchanged.

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

    prior_params: any parameters supplied here overriders the founder_prior_params.

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

    model_meta["name"] = "founder_BB"
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
        "use_xlatent": use_xlatent
    }

    if prior_params is None:
        prior_params = {}
    else:
        if not all(x in founder_prior_params for x in prior_params):
            up = [x for x in prior_params if x not in founder_prior_params]
            warnings.warn(
                "prior_params contain some unknown parameters not"
                f"found in founder_prior_params: {up}. Discarding them."
            )

            prior_params = {
                key: prior_params.get(key, val) for key,val in founder_prior_params.items()
            }

    prior_params = dict(founder_prior_params, **prior_params)

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
        
        coords["c_side"] = ["ra","la"]

        
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
        nassay_experiments = len(coords["assay_experiment"])

        if ppfu_ratios is not None:
            ppfu_ratios = pm.Data("ppfu_ratios", ppfu_ratios)

        input_props = _input_prior(
            nstrains, obs, fixed_input, repeat_idx, ppfu_ratios
        )

        log2_rfs = _log2_rf_prior(
            strains, pt_ests, prior_params, sd_scale, sens
        )
        
        if use_xlatent:
          x_latent, x_sd = _xprior(coords["serum"], prior_params, x,
                                   model_meta["level_sets"])
        else:
          x_latent = x
        
        # titer and slope of neutralization curve and the associated neutralization
        # sigmoid
        if nsera > 0:
            neut = _titers_prior(
                x_latent,
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
        ct_sd = pm.InverseGamma("ct_sd", mu=prior_params["ct_sd_mean"], 
                                sigma=prior_params["ct_sd_sd"])

        # ---- the only change from models.BB_model -----------------------
        log2_sum_fracs = pt.log2(sum_fracs)
        ct_slope = pm.Deterministic(
            "ct_slope",
            1.0
            + pt.exp(
                pm.Normal(
                    "ct_log_excess",
                    mu=np.log(prior_params["ct_slope_excess"]),
                    sigma=sd_scale * prior_params["ct_slope_excess_sd"],
                )
            ),
        )
        if N["NO SERUM"] > 0:
            ref = log2_sum_fracs[: N["NO SERUM"]].mean()
        else:
            logger.info(
                "No NO SERUM samples found. Anchoring the ct slope on the "
                "mean of log2 sum_fracs instead."
            )
            ref = log2_sum_fracs.mean()
        ct_mu = ref + ct_slope * (log2_sum_fracs - ref)

        # founder sampling on the ct WIDTH only. The well's total is a sum over
        # its S surviving founders, whose individual yields are very unequal, so
        # var(log2 total) = k^2/S rather than a constant. Measured directly in
        # the NO SERUM wells - no serum, no ct, no fitted model - the law holds
        # over a 300x range of founder number, with k = 2.10 - 3.97 depending on
        # experiment batch. The counts likelihood is deliberately left alone:
        # the same law applied to the BetaBinomial concentration made the
        # posterior multimodal on the 134 variant panel.
        surv = (extended_neut * input_props).sum(axis=-1)
        nfounders = input_total_pfus * surv
        founder_k = pm.TruncatedNormal(
            "founder_k",
            mu=prior_params["founder_k_mu"],
            sigma=sd_scale * prior_params["founder_k_sd"],
            lower=0.5,
        )
        ct_sigma = pt.sqrt(ct_sd**2 + founder_k**2 / (nfounders + 1e-12))
        # -----------------------------------------------------------------

        pm.Normal(
            "log2_sum_fracs_obs",
            ct_mu,
            ct_sigma,
            observed=obs["ct"],
            dims="assay_sample",
        )
        
        
        
        if use_xlatent:
          pm.Normal("dilution", x_latent, x_sd, observed=x)

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
        
        coords["c_side"] = ["ra","la"]

        
    elif concentration_type == "constant":
        initvals.update(
            {
                "log_conc": np.ones((len(coords["assay_experiment"]),))
                * prior_params["log_conc_mu"]
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
        ct_sd = pm.InverseGamma("ct_sd", mu=prior_params["ct_sd_mean"], 
                                sigma=prior_params["ct_sd_sd"])
        pm.Normal(
            "log2_sum_fracs_obs",
            pt.log2(sum_fracs),
            ct_sd,
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

def _xprior(sera, prior_params, x, level_sets):
  '''
  This goes through so much grief because of the possibility that 
  different sera might have different dilutions or number of repeats.
  Otherwise x_noise and x_sd would be an array with dimension
  ndilution x nsera x nrepeats, cumulatively summed on first axis
  to reflect noise in serial fold dilution setup.
  '''
  
  # SERUM, REPEAT, DIL
  sd = prior_params["x_likelihood_sd"]
  sera_x=\
    [pm.math.cumsum(pm.Normal(f"{sr}_x_noise", 0, prior_params["x_prior_sd"], 
                              dims=[f"{sr}_dilution",f"{sr}_repeat"]), axis=0)
     for sr in sera]
    
  x_var = [pm.math.cumsum(sd**2*pt.ones((len(level_sets[f"{sr}_DILUTION"]), 
                                  len(level_sets[f"{sr}_REPEAT"]))), axis=0)
          for sr in sera]
  
  x_noise = pt.concatenate([pt.flatten(s.T) for s in sera_x])
  x_sd = pt.sqrt(pt.concatenate([pt.flatten(s.T) for s in x_var]))
  
  return x + x_noise, x_sd

