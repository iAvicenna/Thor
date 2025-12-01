#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=bad-indentation, import-error, wrong-import-position, invalid-name, disable=unnecessary-lambda, disable=unsubscriptable-object
# pylint cant see that some pytensor objects are scriptable

"""
Created on Sun Aug 17 18:35:09 2025

@author: avicenna
"""

from typing import List, Dict
import arviz as az
import pymc as pm
import numpy as np
import pytensor.tensor as pt

from .models import _get_counts, _extend_neut, default_prior_params
from .utils import UnacceptableInput


def BB_sample_total_neut(model:pm.model.core.Model,
                         idata:az.data.inference_data.InferenceData,
                         meta:dict) -> None:
  '''
  Used for sampling total_neut and some other related observables concentration.
  Useful for diagnostic purposes and comparing log_S to concentration
  (see paper section Relation Between Noise, PFU and Average Neutralization)
  in the paper.
  '''

  coords = model.coords
  N = _get_counts(meta["processed_table"])
  nstrains = len(coords["strain"])

  prior_params = meta["updated_prior_params"]
  x = idata["constant_data"]["x"].data
  serum_idx = idata["constant_data"]["serum_idx"].data

  if prior_params is None:
    prior_params = {}

  titers = model.log2_titers
  slopes_offset = model.slope_offsets
  slopes = prior_params["s_offset_neut"] + slopes_offset

  mu = (x[:,None] - titers[serum_idx, :])*slopes[serum_idx,None]
  neut = 1 - pm.math.sigmoid(mu) # sigmoid(x)=1/(1+exp(-x))

  extended_neut = _extend_neut(neut, N, nstrains)

  intcpts = model.conc_intercepts

  pm.Deterministic("conc_slope",
                   (intcpts[:,1] - intcpts[:,0])/np.abs(prior_params["prop_threshold"]))


  pm.Deterministic("total_neut", extended_neut.sum(axis=-1),
                   dims=["assay_sample"])


def BB_sample_neut(model:pm.model.core.Model,
                   idata:az.data.inference_data.InferenceData,
                   sera:List[str], strains:List[str], meta:dict) -> None:
  '''
  Used for sampling neutralization curves and latent neutralized pfus from a
  given model and inference object. sera and strains should be supplied.
  This can be memory intensive if chains are very long so you might want to
  shed your chains in such a case. After calling this function you must use
  pm.sample_posterior_predictive on the same model and idata setting
  predictions=True and var_names including neut, neut_inverse.

  neut_inverse is the latent neutralized pfus per variant constructed using the
  sequencing counts, input fractions, and some of the model's fitted parameters.
  See section Fitted Curve Examples in the paper.
  '''
  coords = model.coords
  obs = meta["obs"]
  N = _get_counts(meta["processed_table"])
  nstrains = len(coords["strain"])

  dim1 = [x for x in model.coords["assay_sample"] if
          any(y in x for y in sera + ["NO SERUM"])]

  model.add_coord("neut_sample_dim1", dim1)
  model.add_coord("neut_sample_dim2",  strains)

  Icol = [coords["strain"].index(x) for x in strains]

  prior_params = meta["model_args"]["prior_params"]
  x = idata["constant_data"]["x"].data
  serum_idx = idata["constant_data"]["serum_idx"].data

  if prior_params is None:
    prior_params = {}

  prior_params = dict(default_prior_params, **prior_params)

  if N["INPUT"]>1:
    input_props = pm.Dirichlet("input_props", a=10*np.ones((nstrains)),
                               dims="strain")
  else:
    input_props = obs["input_props"]

  log2_rfs = model.log2_rfs
  titers = model.log2_titers
  slopes_offset = model.slope_offsets
  slopes = prior_params["s_offset_neut"] + slopes_offset

  mu = (x[:,None] - titers[serum_idx, :])*slopes[serum_idx,None]
  neut = 1 - pm.math.sigmoid(mu) # sigmoid(x)=1/(1+exp(-x))

  rfs = 2**log2_rfs

  expanded_neut = _extend_neut(neut, N, nstrains)

  #transformed priors
  fracs = expanded_neut*rfs*input_props
  sum_fracs = fracs.sum(axis=-1)

  obs_props = obs["assay"]/obs["assay"].sum(axis=-1)[:,None]

  neut_inverse = obs_props*sum_fracs[:,None]/(rfs[None,:]*input_props)

  #subselect according to sera
  Irow = [indx for indx,x in enumerate(coords["assay_sample"]) if
          "NO SERUM" in x or any(y in x for y in sera)]

  pm.Deterministic("neut", expanded_neut[np.ix_(Irow,Icol)],
                   dims=["neut_sample_dim1","neut_sample_dim2"])
  pm.Deterministic("neut_inverse", neut_inverse[np.ix_(Irow,Icol)],
                   dims=["neut_sample_dim1","neut_sample_dim2"])


def sample_pairwise_differences(model:pm.model.core.Model)->None:
  '''
  used for sampling for all pairwise differences between titers and
  replicative fitness. Can be memmory intensive.
  '''

  strains = model.coords["strain"]

  log2_rfs = model.log2_rfs
  log2_titers = model.log2_titers

  model.add_coord("strain_col", strains)
  model.add_coord("strain_row", strains)

  pm.Deterministic("dif_log2_rfs", log2_rfs[:,None] - log2_rfs[:,None].T,
                   dims=["strain_row","strain_col"])

  dif_log2_titers = pt.transpose(log2_titers[None,...] -
                                 log2_titers[None,...].T,
                                 (1,0,2))
  pm.Deterministic("dif_log2_titers", dif_log2_titers,
                   dims=["serum","strain_row", "strain_col"])


def sample_relative_titer_differences(model:pm.model.core.Model,
                                      serum_to_ref:dict)->None:
  '''
  samples log2_titers relative to a given variant. use serum_to_ref
  to indicate which reference strain to use for each serum.
  '''

  strains = model.coords["strain"]
  sera = model.coords["serum"]

  if not all(ref in strains for ref in serum_to_ref.values()):
    raise UnacceptableInput("Some serum reference strains are not in the inference data.")

  log2_titers = model.log2_titers

  serum_to_ind = {serum:strains.index(ref) for serum,ref
                  in serum_to_ref.items()}

  Iser = (list(range(len(sera))),
          [serum_to_ind[s] for s in sera])

  ref_titers = log2_titers[Iser]

  # note that in the model highest titers such as 5120
  # has the lowest value
  pm.Deterministic("rel_log2_titers",
                   -(log2_titers - ref_titers[:,None]),
                   dims=["serum","strain"])


def sample_relative_rf_differences(model:pm.model.core.Model,
                                   rf_ref:str)->None:
  '''
  same as sample_relative_titer_differences but for replicative fitness
  and with reference to strain given by rf_ref
  '''

  strains = model.coords["strain"]
  if not rf_ref in strains:
    raise UnacceptableInput("rf reference strain is not in the inference data.")

  log2_rfs = model.log2_rfs
  i0 = strains.index(rf_ref)
  pm.Deterministic("rel_log2_rfs",
                   log2_rfs - log2_rfs[i0],
                   dims=["strain"])


def sample_escape(model:pm.model.core.Model,
                  rank_sera:List[str],
                  serum_to_ref:Dict[str,str]=None)->None:

  '''
  samples escape with respect to sera given in rank_sera. Escape
  is fold drop from the variant given serum_to_ref if not None else
  to the first variant
  '''

  log2_titers = model.log2_titers
  #highest titer has the value 0 as a x-axis covariate
  sera = model.coords["serum"]
  strains = list(model.coords["strain"])

  Iser = [sera.index(s) for s in rank_sera if s in sera]
  if len(Iser)==0:
    raise UnacceptableInput(f"None of rank_sera {rank_sera} are in the model sera {sera}.")

  if len(Iser)>1:
    rank = -log2_titers[Iser,:]
    rank = rank[:,0][:,None] - rank
    rank = rank.mean(axis=0)
    pm.Deterministic("serum_mean_escape_rank", rank,
                     dims=["strain"])

  for serum in rank_sera:
    if serum not in sera:
      continue

    if serum_to_ref is None:
      i0 = 0
    else:
      i0 = strains.index(serum_to_ref[serum])

    Iser = sera.index(serum)

    rank = -log2_titers[Iser,:]
    rank = rank[i0] - rank
    pm.Deterministic(f"{serum}_escape_rank", rank,
                     dims=["strain"])
