from pyspark.sql.types import *

from ..pyspark_test import PySparkTest
from src.main.imputation.median_imputer import MedianImputer


class TestImputeMedian(PySparkTest):
    def test_no_missing_vals(self):
        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (3.0,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        data_expected = [(1.0, 1.0), (3.0, 3.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_one_val(self):
        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (1.0,), (1.0,), (None,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        data_expected = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (None, 1.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_overwrite_column(self):
        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (None,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True)])
        data_expected = [(1.0,), (1.0,)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_even_num_non_nulls(self):
        """
        Notes
        -----
        After reviewing the documentation for how median is selected, the procedure does not
        actually find the true median. There is a relative error allowed so that it gets it
        to be "close enough". So in this test, the result, mathematically, should be 3.5
        filling the null, but instead it is 3.0 .

        """

        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (3.0,), (5.0,), (None,), (4.0,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        data_expected = [(1.0, 1.0), (3.0, 3.0), (5.0, 5.0), (None, 3.0), (4.0, 4.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_odd_num_non_nulls(self):
        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a"]
        self.imputer.output_cols = ["out_a"]

        schema = StructType([StructField("a", FloatType(), True)])
        data = [(1.0,), (3.0,), (5.0,), (None,)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([StructField("a", FloatType(), True), StructField("out_a", FloatType(), True)])
        data_expected = [(1.0, 1.0), (3.0, 3.0), (5.0, 5.0), (None, 3.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_two_col(self):
        self.imputer.strategy = "median"

        self.imputer.input_cols = ["a", "b"]
        self.imputer.output_cols = ["out_a", "out_b"]

        schema = StructType([StructField("a", FloatType(), True), StructField("b", FloatType(), True)])
        data = [(1.0, 2.0),
                (3.0, 4.0),
                (5.0, None),
                (None, 6.0)]
        df = self.spark.createDataFrame(data, schema=schema)

        schema_expected = StructType([
            StructField("a", FloatType(), True),
            StructField("b", FloatType(), True),
            StructField("out_a", FloatType(), True),
            StructField("out_b", FloatType(), True)
        ])

        data_expected = [(1.0, 2.0, 1.0, 2.0),
                         (3.0, 4.0, 3.0, 4.0),
                         (5.0, None, 5.0, 4.0),
                         (None, 6.0, 3.0, 6.0)]
        expected_df = self.spark.createDataFrame(data_expected, schema=schema_expected)

        new_df = self.imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))

    def test_group_by_median(self):
        median_imputer = MedianImputer(input_cols=["Salary"], output_cols=["SalaryImputed"],
                                       list_group_by_cols=[["Job"]])

        data = [("Cashier", 30000.0),
                ("Cashier", None),
                ("Cashier", 37000.0),
                ("Data Scientist", 75000.0),
                ("Data Scientist", 95000.0),
                ("Data Scientist", None),
                ("Cashier", 34000.0),
                ("Data Scientist", 84000.0)]
        df = self.spark.createDataFrame(data, ["Job", "Salary"])

        data_expected = [("Cashier", 30000.0, 30000.0),
                         ("Cashier", None, 34000.0),
                         ("Cashier", 37000.0, 37000.0),
                         ("Cashier", 34000.0, 34000.0),
                         ("Data Scientist", 75000.0, 75000.0),
                         ("Data Scientist", 95000.0, 95000.0),
                         ("Data Scientist", None, 84000.0),
                         ("Data Scientist", 84000.0, 84000.0)]

        expected_df = self.spark.createDataFrame(data_expected, ["Job", "Salary", "SalaryImpute"])

        new_df = median_imputer.impute(df)
        self.assertTrue(self.compare_df(expected_df, new_df))
