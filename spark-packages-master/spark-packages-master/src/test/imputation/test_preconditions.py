"""
General testing structure taken from:
    https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
"""
from pyspark.sql.types import *
from ..pyspark_test import PySparkTest


class TestImputationPreconditions(PySparkTest):

    def test_impute_preconditions(self):
        # input/output column lengths mismatch
        schema = StructType([
            StructField("in_col1", StringType(), True),
            StructField("in_col2", StringType(), True),
        ])

        data = [("data11", "data12"), ("data21", "data22")]
        df = self.spark.createDataFrame(data, schema=schema)

        with self.assertRaises(Exception) as e:
            self.imputer.input_cols = ["in_col1", "in_col2"]
            self.imputer.output_cols = ["out_col1"]
            new_df = self.imputer.impute(df)

        self.assertTrue("must match length" in str(e.exception))


