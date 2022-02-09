import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionSummary

from ..pyspark_test import PySparkTest
from src.main.imputation.linear_regression_imputer import LinearRegressionImputer


class TestLRI(PySparkTest):

    def test_simple_lin_regression(self):
        lr_imputer = LinearRegressionImputer(features_col="features", col_to_impute="y")

        data = [
            [1.57, 1],
            [3.03, 2],
            [4.55, 3],
            [None, 4],
            [7.6, 5],
            [9.04, 6],
            [10.53, 7],
            [None, 8],
            [13.58, 9]]
        df = self.spark.createDataFrame(data, ["y", "x"])
        df = VectorAssembler(inputCols=["x"], outputCol="features") \
            .transform(df)

        expected_data = [
            [1.57, 1],
            [3.03, 2],
            [4.55, 3],
            [6.005, 4],
            [7.6, 5],
            [9.04, 6],
            [10.53, 7],
            [12.010, 8],
            [13.58, 9]]
        df_expected = self.spark.createDataFrame(expected_data, ["y", "x"])

        # dropping features just for testing purposes
        # need to cast to double since the dot_prod udf is casting to float to avoid py4j errors.
        # need to round to handle floating point errors.
        new_df = lr_imputer.impute(df) \
            .withColumn("y",
                        F.round(F.col("y").cast("double"), 3)) \
            .drop("features")

        self.assertTrue(self.compare_df(new_df, df_expected))

    def test_multivariate_lin_regression(self):
        lr_imputer = LinearRegressionImputer(features_col="features", col_to_impute="Advertising (millions)")

        data = [
            (1, 651, 23),
            (2, 762, 26),
            (3, 856, 30),
            (4, 1063, None),
            (5, 1190, 43),
            (6, 1298, 48),
            (7, 1421, None),
            (8, 1440, 57),
            (9, 1518, 58)]
        df = self.spark.createDataFrame(data, ["Year", "Sales (millions)", "Advertising (millions)"])
        df = VectorAssembler(inputCols=["Year", "Sales (millions)"], outputCol="features") \
            .transform(df)

        data_expected = [
            (1, 651, 23.0),
            (2, 762, 26.0),
            (3, 856, 30.0),
            (4, 1063, 36.8376),
            (5, 1190, 43.0),
            (6, 1298, 48.0),
            (7, 1421, 51.7155),
            (8, 1440, 57.0),
            (9, 1518, 58.0)]
        df_expected = self.spark.createDataFrame(data_expected, ["Year", "Sales (millions)", "Advertising (millions)"])

        # dropping features just for testing purposes
        # need to cast to double since the dot_prod udf is casting to float to avoid py4j errors.
        # need to round to handle floating point errors.
        new_df = lr_imputer.impute(df) \
            .withColumn("Advertising (millions)",
                        F.round(F.col("Advertising (millions)").cast("double"), 4)) \
            .drop("features")

        self.assertTrue(self.compare_df(new_df, df_expected))

    def test_storing_summary_stats(self):
        lr_imputer = LinearRegressionImputer(features_col="features", col_to_impute="y",
                                             store_summary_statistics=False)

        # no summary stats until imputation is performed
        self.assertTrue(lr_imputer.summary_statistics is None)

        data = [
            [1.57, 1],
            [3.03, 2],
            [4.55, 3],
            [None, 4],
            [7.6, 5],
            [9.04, 6],
            [10.53, 7],
            [None, 8],
            [13.58, 9]]

        df = self.spark.createDataFrame(data, ["y", "x"])
        df = VectorAssembler(inputCols=["x"], outputCol="features") \
            .transform(df)

        lr_imputer.impute(df)

        # should still be None after imputation
        self.assertTrue(lr_imputer.summary_statistics is None)

        lr_imputer.store_summary_statistics = True
        lr_imputer.impute(df)

        # summary stats should contain LinearRegressionSummary object now.
        self.assertTrue(isinstance(lr_imputer.summary_statistics, LinearRegressionSummary))
