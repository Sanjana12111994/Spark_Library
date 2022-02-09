from typing import List

from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer
import pyspark.sql.functions as F

from .imputer_params import HasInputOutputCols, HasMissingValue, HasRelativeError, HasListGroupByCols
from .utils import warn_if_cols_have_nulls


class MedianImputer(HasInputOutputCols, HasMissingValue, HasRelativeError, HasListGroupByCols):
    """
    An imputation class to impute by median either for a whole column, or by subgroups.

    Parameters
    ----------
    input_cols : List[str]
        A list of column names with missing values to be imputed.
    output_cols : List[str]
        A list of column names where for index i in 0, ..., n-1 for
        input_cols of length n, input_cols[i] -> output_cols[i].
        Defaults to equaling input_cols and overwrites the input columns.
    missing_value : float
        A specified value to be imputed in addition to `null`.
    relative_error : float
        The margin of error allowed for imputing by the median.
        Defaults to 0.001 as per pyspark.ml.feature.Imputer.
    list_group_by_cols : List[List[str]]
        A List of List of column names at index i in 0, ..., n-1
        to do imputation by subgroups on. Must be same length as `input_cols`.

    See Also
    --------
    HasInputOutputCols, HasMissingValue, HasRelativeError, HasListGroupByCols

    Notes
    -----
    If `list_group_by_cols` is not specified:
        1. behaviour defaults to simply using Imputer from pyspark.ml.feature.
        TODO: look into a way to get around this problem. Maybe check if column is numeric and then cast?
        2. each input column must be of dtype Float or Double so that Imputer doesn't throw an error.

    If `list_group_by_cols` is specified:
        1. If there are nulls in any of the group by columns, they will be their own group category.
           This behaviour may be useful, but specific domain knowledge will be critical to doing so.
           If in doubt, impute your group by columns first, before using `MedianImputer`.
        2. `missing_value` is not used.     TODO: implement the correct functionality
        3. `relative_error` is not used.    TODO: implement the correct functionality

    Single imputation methods treat the imputed value as the "true" value, which does not account for any
    uncertainty in the imputation, often resulting in variance being underestimated as well as an
    overestimation of precision.

    References
    ----------
    [1] Donders et al. (2006). Review: A gentle introduction to imputation of missing values.
        Journal of Clinical Epidemiology, 59(10), 1087-1091.

    Examples
    --------
    >>> df = spark.createDataFrame([(1.0,), (float("nan"),), (2.0,), (4.0,)], ["a"])
    >>> median_imputer = MedianImputer(input_cols=["a"])
    >>> median_imputer.impute(df).show()
    +---+
    |  a|
    +---+
    |1.0|
    |2.0|
    |2.0|
    |4.0|
    +---+

    >>> df = spark.createDataFrame([("Cashier", 30000.0), ("Cashier", None), ("Cashier", 37000.0), \
        ("Data Scientist", 75000.0), ("Data Scientist", 95000.0), ("Data Scientist", None), ("Cashier", 34000.0), \
        ("Data Scientist", 84000.0)], ["Job", "Salary"])
    >>> median_imputer = MedianImputer(input_cols=["Salary"], output_cols=["SalaryImputed"], \
                                       list_group_by_cols=[["Job"]])
    >>> median_imputer.impute(df).show()
    +--------------+-------+-------------+
    |           Job| Salary|SalaryImputed|
    +--------------+-------+-------------+
    |       Cashier|30000.0|      30000.0|
    |       Cashier|   null|      34000.0|
    |       Cashier|37000.0|      37000.0|
    |       Cashier|34000.0|      34000.0|
    |Data Scientist|75000.0|      75000.0|
    |Data Scientist|95000.0|      95000.0|
    |Data Scientist|   null|      84000.0|
    |Data Scientist|84000.0|      84000.0|
    +--------------+-------+-------------+

    """
    def __init__(self,
                 input_cols: List[str] = None,
                 output_cols: List[str] = None,
                 missing_value: float = float("nan"),
                 relative_error: float = 0.001,
                 list_group_by_cols: List[List[str]] = None):

        HasInputOutputCols.__init__(self, input_cols=input_cols, output_cols=output_cols)
        HasMissingValue.__init__(self, missing_value=missing_value)
        HasRelativeError.__init__(self, relative_error=relative_error)
        HasListGroupByCols.__init__(self, list_group_by_cols=list_group_by_cols)

    def impute(self, df: DataFrame) -> DataFrame:
        if len(self.input_cols) != len(self.output_cols):
            raise ValueError("Length of input columns must match length of output columns.")

        if self.list_group_by_cols is None:
            # Imputer requires the inputCols to be of type Float or Double.
            imputer = Imputer(strategy="median",
                              inputCols=self.input_cols,
                              outputCols=self.output_cols,
                              missingValue=self.missing_value)
            df = imputer.fit(df).transform(df)

        else:
            if len(self.list_group_by_cols) != len(self.input_cols):
                raise ValueError("Length of List group by cols must match length of input columns.")

            for index in range(len(self.input_cols)):
                in_col, out_col, group_by_cols = self.input_cols[index], self.output_cols[index], \
                                                 self.list_group_by_cols[index]

                # warn_if_cols_have_nulls(df, group_by_cols)

                # Taken from:
                # https://stackoverflow.com/questions/46845672/median-quantiles-within-pyspark-groupby
                percentile_approx = F.expr("percentile_approx({}, 0.5)".format(in_col))

                group_by_medians = df.groupBy(group_by_cols).agg(percentile_approx.alias("group_median"))
                df_with_medians = df.join(group_by_medians, on=group_by_cols)

                df = df_with_medians \
                    .withColumn(out_col, F.coalesce(in_col, "group_median")) \
                    .drop("group_median")

        return df
