from pyspark.sql.types import *

from ..pyspark_test import PySparkTest
from src.main.imputation.forward_fill_imputer import ForwardFillImputer


class TestFFillImpute(PySparkTest):
    # global variables are hacky, I apologize.
    global schema, data
    schema = StructType([
            StructField("date", IntegerType()),
            StructField("location", StringType()),
            StructField("temperature", IntegerType()),
            StructField("unit", StringType())])

    data = [
        (1, 'a', None, None),
        (3, 'a', None, None),
        (2, 'a', 45, "C"),
        (4, 'a', 40, "F"),
        (3, 'b', None, None),
        (2, 'b', 25, "F"),
        (4, 'b', 20, "C"),
        (1, 'b', None, None)]

    def test_ffill_single(self):
        ffill_imputer = ForwardFillImputer(input_cols=["temperature"], output_cols=["temperature"],
                                           order_by_cols=["date"], list_partition_by_cols=[["location"]])

        df_input = self.spark.createDataFrame(data, schema=schema)

        data_expected = [(1, 'a', None, None),
                         (1, 'b', None, None),
                         (2, 'a', 45, "C"),
                         (2, 'b', 25, "F"),
                         (3, 'a', 45, None),
                         (3, 'b', 25, None),
                         (4, 'a', 40, "F"),
                         (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_expected, schema=schema)

        imputed_df = ffill_imputer.impute(df_input).orderBy("date", "location")
        self.assertTrue(self.compare_df(df_expected, imputed_df))
       
    def test_ffill_multiple(self):
        ffill_imputer = ForwardFillImputer(input_cols=["temperature", "unit"], order_by_cols=["date", "date"],
                                           list_partition_by_cols=[["location"], ["location"]])

        df_input = self.spark.createDataFrame(data, schema=schema)
        
        data_exp = [
            (1, 'a', None, None),
            (1, 'b', None, None),
            (2, 'a', 45, "C"),
            (2, 'b', 25, "F"),
            (3, 'a', 45, "C"),
            (3, 'b', 25, "F"),
            (4, 'a', 40, "F"),
            (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_exp, schema=schema)

        imputed_df = ffill_imputer.impute(df_input).orderBy("date", "location")
        self.assertTrue(self.compare_df(df_expected, imputed_df))

    def test_ffill_no_partition(self):
        ffill_imputer = ForwardFillImputer(input_cols=["temperature", "unit"], order_by_cols=["date", "date"])

        df_input = self.spark.createDataFrame(data, schema=schema)

        data_exp = [
            (1, 'a', None, None),
            (1, 'b', None, None),
            (2, 'a', 45, "C"),
            (2, 'b', 25, "F"),
            (3, 'a', 25, "F"),
            (3, 'b', 25, "F"),
            (4, 'a', 40, "F"),
            (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_exp, schema=schema)

        imputed_df = ffill_imputer.impute(df_input).orderBy("date")
        self.assertTrue(self.compare_df(df_expected, imputed_df))
