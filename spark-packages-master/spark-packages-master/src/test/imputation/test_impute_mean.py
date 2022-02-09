from pyspark.sql.types import *

from ..pyspark_test import PySparkTest
from src.main.imputation.mean_imputer import MeanImputer


class TestImputeMean(PySparkTest):

    def test_no_missing_val(self):
        self.imputer.strategy = "mean"

        # no missing values
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (1.0,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        data_expected = [(1.0, 1.0), (1.0, 1.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_one_val(self):
        # only one value to get mean of
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        data = [(1.0,), (None,)]
        schema = StructType([StructField("a", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1.0, 1.0), (None, 1.0)]
        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_overwrite_col(self):
        # overwrite original column
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = None  # output_cols = input_cols

        data = [(1.0,), (None,)]
        schema = StructType([StructField("a", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1.0,), (1.0,)]
        schema_expected = schema
        df_expected = self.spark.createDataFrame(data_expected, schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(df_expected, new_df))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def test_multiple_vals(self):
        # more than one value to average
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["a"]

        data = [(1.0,), (2.0,), (None,)]
        schema = StructType([StructField("a", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1.0,), (2.0,), (1.5,)]
        schema_expected = schema
        df_expected = self.spark.createDataFrame(data_expected, schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(df_expected, new_df))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def test_two_col(self):
        # two columns
        self.imputer.input_cols = ["a", "b"]
        self.imputer.output_cols = ["a", "b"]

        data = [(1.0, 4.0), (2.0, None), (None, 8.0)]
        schema = StructType([StructField("a", FloatType(), True), StructField("b", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1.0, 4.0), (2.0, 6.0), (1.5, 8.0)]
        schema_expected = schema
        df_expected = self.spark.createDataFrame(data_expected, schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(df_expected, new_df))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def test_impute_only_one_col(self):
        # two columns, impute one
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["a"]

        data = [(1.0, 4.0), (2.0, None), (None, 10.0)]
        schema = StructType([StructField("a", FloatType(), True), StructField("b", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1.0, 4.0), (2.0, None), (1.5, 10.0)]
        schema_expected = schema
        df_expected = self.spark.createDataFrame(data_expected, schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(df_expected, new_df))

    def test_col_is_null(self):
        # column missing all values
        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        data = [(None, 2.0), (None, 3.5), (None, 9.3)]
        schema = StructType([StructField("a", FloatType(), True), StructField("b", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        # no expected data needed since this raises an Exception
        with self.assertRaises(Exception) as e:
            new_df = self.imputer.impute(df)

        self.assertIn("All the values in a are Null, Nan or missingValue(NaN)", str(e.exception))

    def test_group_by_mean(self):
        mean_imputer = MeanImputer(input_cols=["salary"], list_group_by_cols=[["occupation"]])
        # impute with different means for different groups
        data = [
            ["cashier", 30000.0],
            ["cashier", None],
            ["cashier", 37000.0],
            ["data scientist", 75000.0],
            ["data scientist", 95000.0],
            ["data scientist", None]]
        schema = StructType([StructField("occupation", StringType(), True), StructField("salary", FloatType(), True)])
        df = self.spark.createDataFrame(data, schema=schema)

        data_expected = [
            ["data scientist", 75000.0],
            ["data scientist", 95000.0],
            ["data scientist", 85000.0],
            ["cashier", 30000.0],
            ["cashier", 33500.0],
            ["cashier", 37000.0]]
        schema_expected = schema
        df_expected = self.spark.createDataFrame(data_expected, schema_expected)

        new_df = mean_imputer.impute(df)

        # imputation returns salary as float64 instead of float32 when it's originally created.
        # this is a temporary fix which works as intended.
        self.assertTrue(self.compare_df(df_expected, new_df))
