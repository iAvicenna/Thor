#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:09:55 2025

@author: avicenna

Some basic tests which only check that stuff such as table formatting etc
are done correctly and exceptions for bad input are caught correctly.
Does not do full model run tests as that would be too lengthy. It does
however check that models compile OK.
"""

import unittest
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import Thor.models as tm

_CWD = Path(__file__).parent.resolve()
_DATA = Path(_CWD, "test_inputs")

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def _check_table_eq(table1, table2):

    return1 = all(table1.index == table2.index)
    return2 = all(table2.index == table2.index)
    return3 = np.all(table1.values == table2.values)

    return return1 and return2 and return3


class TablePreprocessing(unittest.TestCase):

    def setUp(self):

        self.table = pd.read_csv(
            Path(_DATA, "ngs_table.csv"), index_col=[0, 1, 2], header=0
        )
        self.table = tm._preprocess_table(self.table)
        self.strains = [x for x in self.table.columns if x != "CT"]

        seed = np.random.SeedSequence().spawn(1)[0]
        self.rng = np.random.default_rng(seed)

        self.model_meta = {}
        self.model_meta = {"model_args": {"input_total_pfus": 10000}}
        self.model_meta["processed_table"] = self.table
        # run this so model_meta gets factor_table which is used in tests
        tm._factorize_table(self.model_meta, self.table, False)

    def test_preprocess(self):

        I = self.rng.permutation(range(self.table.shape[0]))
        table_shuf = tm._preprocess_table(self.table.iloc[I, :].copy())
        self.assertTrue(_check_table_eq(self.table, table_shuf))

    def test_factorize(self):

        I = self.rng.permutation(range(self.table.shape[0]))
        table = tm._preprocess_table(self.table.iloc[I, :].copy())

        model_meta = {"model_args": {"input_total_pfus": 10000}}
        model_meta["processed_table"] = table

        tm._factorize_table(model_meta, table, False)

        ordered_dil_cov = list(range(8, 0)) * 4
        self.assertTrue(
            all(
                x == y
                for x, y in zip(
                    ordered_dil_cov, model_meta["dilution_covariates"]
                )
            )
        )

        # conversion to str for comparison of nans
        self.assertTrue(
            _check_table_eq(
                model_meta["factor_table"].astype(str),
                self.model_meta["factor_table"].astype(str),
            )
        )

        self.assertDictEqual(
            model_meta["level_sets"], self.model_meta["level_sets"]
        )

        self.assertTrue(
            np.all(model_meta["strains"] == self.model_meta["strains"])
        )

    def test_exceptions(self):

        # test no serum in table
        table = self.table.copy()
        table.index = table.index.droplevel("SERUM")

        with self.assertRaises(tm.BadTableFormat):
            tm._preprocess_table(table)

        # test bad dilution format
        table = self.table.copy()
        index = list(table.index)
        index[10] = ("F09012", "A", "2560")
        table.index = pd.MultiIndex.from_tuples(index)
        table.index.names = ["SERUM", "REPEAT", "DILUTION"]

        with self.assertRaises(tm.BadDilutionFormat):
            tm._preprocess_table(table)

        # test wrong input repeats when fixed_input=True
        table = self.table.copy()
        I0 = [x for x in table.index if x[0] == "INPUT"][0]
        table.loc[("INPUT", "A", "")] = table.loc[I0, :]
        table.loc[("INPUT", "C", "")] = table.loc[I0, :]
        table.drop(I0, inplace=True)

        model_meta = {}
        model_meta = {"model_args": {"input_total_pfus": 10000}}
        model_meta["processed_table"] = table

        with self.assertRaises(tm.BadTableFormat):
            tm._factorize_table(model_meta, table, True)

        # test uncompatible dilutions
        table = self.table.copy()
        table.loc[("F09012", "A", "1/5"), :] = table.loc[
            ("F09012", "A", "1/20"), :
        ]
        table.drop(("F09012", "A", "1/20"), inplace=True)

        model_meta = {}
        model_meta = {"model_args": {"input_total_pfus": 10000}}
        model_meta["processed_table"] = table
        tm._factorize_table(model_meta, table, False)

        with self.assertRaises(tm.BadTableFormat):
            tm.get_pt_ests(model_meta)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.table = pd.read_csv(
            Path(_DATA, "ngs_table.csv"), index_col=[0, 1, 2], header=0
        )

    def test_compile(self):
        # make sure models construct and likelihoods eval wo failing
        # warning: model debug may fail since there are Truncated distributions
        # to which initial points are supplied when sampling

        ppfu_ratios = list(np.ones((self.table.shape[1] - 1)))

        for concentration_type in ["parametric"]:
            with self.subTest(f"BB_model {concentration_type}"):

                model, _ = tm.BB_model(
                    self.table,
                    10000,
                    concentration_type=concentration_type,
                    ppfu_ratios=ppfu_ratios,
                )

                model.eval_rv_shapes()
                model.counts.eval()
                model.log2_sum_fracs_obs.eval()

                if hasattr(model, "input_counts"):
                  model.input_counts.eval()

            with self.subTest(f"BB_mix_model {concentration_type}"):

                model, _ = tm.BB_mix_model(
                    self.table,
                    10000,
                    concentration_type=concentration_type,
                    ppfu_ratios=ppfu_ratios,
                )

                model.eval_rv_shapes()
                model.counts.eval()
                model.log2_sum_fracs_obs.eval()

                if hasattr(model, "input_counts"):
                  model.input_counts.eval()


if __name__ == "__main__":

    unittest.main()
