import unittest
import logging

from pyspark.sql import SparkSession

from src.main.imputation.simple_imputer import SimpleImputer


class PySparkTest(unittest.TestCase):

    @classmethod
    def quiet_py4j(cls):
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @classmethod
    def create_test_pyspark_session(cls):
        return SparkSession.builder\
                           .appName("pyspark-local-testing")\
                           .getOrCreate()

    @classmethod
    def compare_df(cls, df1, df2):
        """
        Notes
        -----
        This doesn't work if there are NaN's in any columns.
        NaN has the annoying property that NaN != NaN.

        """
        return df1.collect() == df2.collect()


    @classmethod
    def setUpClass(cls):
        cls.quiet_py4j()
        cls.spark = cls.create_test_pyspark_session()
        cls.imputer = SimpleImputer(input_cols=["in_col"], output_cols=["out_col"])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


