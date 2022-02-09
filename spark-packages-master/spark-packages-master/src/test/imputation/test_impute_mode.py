#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:12:51 2020
@author: Kevin
"""
from ..pyspark_test import PySparkTest
from src.main.imputation.mode_imputer import ModeImputer


class TestModeImpute(PySparkTest):
    
    ''' 
    Test DF
    +---+----+---+----+
    | id|  c1| c2|  c3|
    +---+----+---+----+
    |  1|   2|null|   0|
    |  2|   2|null|   0|
    |  3|   3|null|   0|
    |  4|   3|null|   0|
    |  5|null|null|null|
    +---+----+---+----+
    '''
    
    # Edge case 1 multimodal column
    def test_multimodal(self):
        """
        Notes
        -----
        What is this actually testing?

        """
        self.imputer.strategy = "mode"
        df = self.spark.createDataFrame([(1, 2, -1, 0), (2, 2, None, 0), (3, 3, None, 0), (4, 3, None, 0), (5, None, None, None)], ("id", "c1", "c2", "c3"))
        expected_df = self.spark.createDataFrame([(1, 2, -1, 0), (2, 2, None, 0), (3, 3, None, 0), (4, 3, None, 0), (5, 3, None, None)], ("id", "c1", "c2", "c3"))
        
        self.imputer.input_cols = ["c1"]
        self.imputer.output_cols = ["c1"]
        
        new_df = self.imputer.impute(df)

        self.assertTrue(self.compare_df(expected_df, new_df))
    
    # Edge case 2 column entirely corrupted
    def test_degenerate(self):
        """
        Notes
        -----
        this test doesn't run as expected. "c2" is all null, and creating a DF from it
        results in ValueError being raised. Should this be an expected behaviour?
        """
        # df = self.spark.createDataFrame([(1, 2, None, 0), (2, 2, None, 0), (3, 3, None, 0), (4, 3, None, 0), (5, None, None, None)], ("id", "c1", "c2", "c3"))
        # expected_df = self.spark.createDataFrame([(1, 2, 0), (2, 2, 0), (3, 3, 0), (4, 3, 0), (5, None, None)], ("id", "c1", "c3"))
        #
        # self.imputer.input_cols(["c2"])
        # self.imputer.output_cols(["c2"])
        #
        # new_df = self.imputer.impute(df)
        # '''
        # Expected Output
        # +---+----+----+
        # | id|  c1|  c3|
        # +---+----+----+
        # |  1|   2|   0|
        # |  2|   2|   0|
        # |  3|   3|   0|
        # |  4|   3|   0|
        # |  5|null|null|
        # +---+----+----+
        # '''
        # self.assertTrue(self.compare_df(expected_df, new_df))

    def test_group_by_mode(self):
        mode_imputer = ModeImputer(input_cols=["FavouriteColour"], list_group_by_cols=[["Country"]])

        data = [("Red", "CA"),
                ("Red", "CA"),
                ("Blue", "CA"),
                ("Blue", "USA"),
                ("Blue", "USA"),
                (None, "CA"),
                (None, "USA")]
        df = self.spark.createDataFrame(data, ["FavouriteColour", "Country"])

        expected_data = [("CA", "Red"),
                         ("CA", "Red"),
                         ("CA", "Blue"),
                         ("CA", "Red"),
                         ("USA", "Blue"),
                         ("USA", "Blue"),
                         ("USA", "Blue")]
        expected_df = self.spark.createDataFrame(expected_data, ["Country", "FavouriteColour"])

        new_df = mode_imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))
