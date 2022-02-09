from typing import List

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler    # will be used when impute_initial param is implemented
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.types import *
import pyspark.sql.functions as F


class LinearRegressionImputer(object):
    """
    Create a linear regression imputer for a single column.

    Parameters
    ----------
    features_col : str
        The column where features are stored in the DataFrame that is to be imputed.
    features_col_names : List[str]    TODO: implement this functionality
        A list of column names which were used to generate the `features_col`. This
        is used when `impute_initial` is set to `True`, since nulls in the features
        must be imputed.
    col_to_impute : str
        The target column to be imputed.
    store_summary_statistics : bool
        Stores the `LinearRegressionSummary` in the `summary_statistics` attribute that is returned
        when a fitted linear regression model calls `evaluate`, as outlined in:
        https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.regression
    impute_initial : bool   TODO: implement this functionality. Should default to True once it works.
        Specifies whether temporary initial imputations are to be done on `features_col_names`.

    Notes
    -----
    When initial imputation is set to False, it is assumed that the feature columns
    do not have any null values present.

    Much like other single imputation methods, Linear Regression Imputation treats the imputed value as
    the "true" value, which does not account for any uncertainty in the imputation, which often results
    in variance being underestimated as well as an overestimation of precision.

    References
    ----------
    [1] Donders et al. (2006). Review: A gentle introduction to imputation of missing values.
        Journal of Clinical Epidemiology, 59(10), 1087-1091.

    Examples
    --------
    >>> df = spark.createDataFrame([(1, 651, 23), (2, 762, 26), (3, 856, 30), (4, 1063, None), (5, 1190, 43), \
        (6, 1298, 48), (7, 1421, None), (8, 1440, 57), (9, 1518, 58)], \
        ["Year", "Sales (millions)", "Advertising (millions)"])
    >>> df = VectorAssembler(inputCols=["Year", "Sales (millions)"], outputCol="features").transform(df)
    >>> lr_imputer = LinearRegressionImputer(features_col="features", col_to_impute="Advertising (millions)")
    >>> lr_imputer.impute(df).show()
    +----+--------------------+--------------------------+------------+
    |Year|Sales (millions USD)|Advertising (millions USD)|    features|
    +----+--------------------+--------------------------+------------+
    |   1|                 651|                      23.0| [1.0,651.0]|
    |   2|                 762|                      26.0| [2.0,762.0]|
    |   3|                 856|                      30.0| [3.0,856.0]|
    |   4|                1063|                   36.8376|[4.0,1063.0]|
    |   5|                1190|                      43.0|[5.0,1190.0]|
    |   6|                1298|                      48.0|[6.0,1298.0]|
    |   7|                1421|                 51.715458|[7.0,1421.0]|
    |   8|                1440|                      57.0|[8.0,1440.0]|
    |   9|                1518|                      58.0|[9.0,1518.0]|
    +----+--------------------+--------------------------+------------+

    """
    # TODO: add option to specify what the output column for the imputation is
    # TODO: consider adding param which specifies what initial imputation strategy to use for each independent variable
    # TODO: add additional params / dict to allow further customization of the LinearRegression used.
    def __init__(self, features_col: str = "features", features_col_names: List[str] = None,
                 col_to_impute: str = "label", store_summary_statistics: bool = False,
                 impute_initial: bool = False,):
        self.features_col = features_col
        self.features_col_names = features_col_names
        self.col_to_impute = col_to_impute
        self.store_summary_statistics = store_summary_statistics
        self.impute_initial = False  # TODO: change this to impute_initial once it's supported
        self.lm = LinearRegression(featuresCol=features_col, labelCol=col_to_impute)
        self.summary_statistics = None

    def impute(self, df: DataFrame) -> DataFrame:
        """
        Impute the DataFrame via Linear Regression Imputation
        """
        # Assumes feature column has no nulls present.
        features_and_label_df = df \
            .select(self.features_col, self.col_to_impute) \
            .na.drop(subset=self.col_to_impute)

        lm = self.lm
        if not self.impute_initial:
            # 2020-08-13
            # trained_model and coefs will always be defined since impute_initial
            # is explicitly defined as False until its functionality is implemented
            trained_model = lm.fit(features_and_label_df)
            coefs = trained_model.coefficients

        if self.store_summary_statistics:
            self.summary_statistics = trained_model.evaluate(features_and_label_df)

        dot_prod_udf = self._dot_prod_udf()
        imputed_df = df \
            .withColumn("coefs", self._vec_col_udf(coefs)) \
            .withColumn(self.col_to_impute,
                        F.when(F.col(self.col_to_impute).isNull(),
                               dot_prod_udf(F.col("coefs"), F.col(self.features_col)))
                        .otherwise(F.col(self.col_to_impute))) \
            .drop("coefs")

        return imputed_df

    def _vec_col_udf(self, vector: DenseVector):
        """
        Return udf for creating a `lit` column of `vector`.
        """
        vec_col_udf = F.udf(lambda: vector, VectorUDT())()
        return vec_col_udf

    def _dot_prod_udf(self):
        """
        Return udf for applying dot product coefs vector and features vector.
        """
        # NOTE:
        # Dot product MUST be cast to float since py4j can not convert numpy.float64 -> float.
        def dot_prod(vec1, vec2):
            return float(vec1.dot(vec2))

        dot_prod_udf = F.udf(lambda vec1, vec2: dot_prod(vec1, vec2), FloatType())
        return dot_prod_udf
