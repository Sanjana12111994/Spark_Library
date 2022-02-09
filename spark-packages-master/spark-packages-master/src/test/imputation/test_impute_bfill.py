from pyspark.sql.types import *

from ..pyspark_test import PySparkTest
from src.main.imputation.backward_fill_imputer import BackwardFillImputer


class TestBFillImpute(PySparkTest):
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

    def test_bfill_single(self):
        bfill_imputer = BackwardFillImputer(input_cols=["temperature"], order_by_cols=["date"],
                                            list_partition_by_cols=[["location"]])

        df_input = self.spark.createDataFrame(data, schema=schema)

        data_exp = [
            (1, 'a', 45, None),
            (1, 'b', 25, None),
            (2, 'a', 45, "C"),
            (2, 'b', 25, "F"),
            (3, 'a', 40, None),
            (3, 'b', 20, None),
            (4, 'a', 40, "F"),
            (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_exp, schema=schema)
        
        imputed_df = bfill_imputer.impute(df_input).orderBy("date", "location")
        self.assertTrue(self.compare_df(df_expected, imputed_df))
       
    # Test Case 2 - Check multiple column imputation for backfill
    def test_bfill_multiple(self):
        bfill_imputer = BackwardFillImputer(input_cols=["temperature", "unit"], order_by_cols=["date", "date"],
                                            list_partition_by_cols=[["location"], ["location"]])

        df_input = self.spark.createDataFrame(data, schema=schema)

        data_exp = [
            (1, 'a', 45, "C"),
            (1, 'b', 25, "F"),
            (2, 'a', 45, "C"),
            (2, 'b', 25, "F"),
            (3, 'a', 40, "F"),
            (3, 'b', 20, "C"),
            (4, 'a', 40, "F"),
            (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_exp, schema=schema)

        imputed_df = bfill_imputer.impute(df_input).orderBy("date", "location")
        
        self.assertTrue(self.compare_df(df_expected, imputed_df))
             
    # Test Case 3 - Check imputation for backfill without partition_by
    def test_bfill_no_partition(self):
        bfill_imputer = BackwardFillImputer(input_cols=["temperature", "unit"], order_by_cols=["date", "date"])

        df_input = self.spark.createDataFrame(data, schema=schema)

        data_exp = [
            (1, 'a', 45, "C"),
            (1, 'b', 45, "C"),
            (2, 'a', 45, "C"),
            (2, 'b', 25, "F"),
            (3, 'a', 40, "F"),
            (3, 'b', 40, "F"),
            (4, 'a', 40, "F"),
            (4, 'b', 20, "C")]
        df_expected = self.spark.createDataFrame(data_exp, schema=schema)
    
        imputed_df = bfill_imputer.impute(df_input).orderBy("date")
        self.assertTrue(self.compare_df(df_expected, imputed_df))
