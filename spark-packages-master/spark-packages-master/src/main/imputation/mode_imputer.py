from typing import List

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

from .imputer_params import HasInputOutputCols, HasListGroupByCols
from .utils import warn_if_cols_have_nulls


class ModeImputer(HasInputOutputCols, HasListGroupByCols):
    """
    An imputation class to impute by mode either for a whole column, or by subgroups.

    Parameters
    ----------
    input_cols : List[str]
        A list of column names with missing values to be imputed.
    output_cols : List[str]
        A list of column names where for index i in 0, ..., n-1 for
        input_cols of length n, input_cols[i] -> output_cols[i].
        Defaults to equaling input_cols and overwrites the input columns.
    list_group_by_cols : List[List[str]]
        A List of List of column names at index i in 0, ..., n-1
        to do imputation by subgroups on. Must be same length as `input_cols`.

    See Also
    --------
    HasInputOutputCols, HasListGroupByCols

    Notes
    -----
    If `list_group_by_cols` is specified:
        1. If there are nulls in any of the group by columns, they will be their own group category.
           This behaviour may be useful, but specific domain knowledge will be critical to doing so.
           If in doubt, impute your group by columns first.

    Single imputation methods treat the imputed value as the "true" value, which does not account for any
    uncertainty in the imputation, often resulting in variance being underestimated as well as an
    overestimation of precision.

    References
    ----------
    [1] Donders et al. (2006). Review: A gentle introduction to imputation of missing values.
        Journal of Clinical Epidemiology, 59(10), 1087-1091.

    Examples
    --------
    >>> df = spark.createDataFrame([("a",), ("a",), ("b",), (None,)], ["letters"])
    >>> mode_imputer = ModeImputer(input_cols=["letters"])
    >>> mode_imputer.impute(df).show()
    +-------+
    |letters|
    +-------+
    |      a|
    |      a|
    |      b|
    |      a|
    +-------+

    >>> df = spark.createDataFrame([("Red", "CA"), ("Red", "CA"), ("Blue", "CA"), ("Blue", "USA"), ("Blue", "USA"), \
                                    (None, "CA"), (None, "USA")], ["FavouriteColour", "Country"])
    >>> mode_imputer = ModeImputer(input_cols=["FavouriteColour"], list_group_by_cols=[["Country"]])
    >>> mode_imputer.impute(df).show()
    +-------+---------------+
    |Country|FavouriteColour|
    +-------+---------------+
    |     CA|            Red|
    |     CA|            Red|
    |     CA|           Blue|
    |     CA|            Red|
    |    USA|           Blue|
    |    USA|           Blue|
    |    USA|           Blue|
    +-------+---------------+

    """
    def __init__(self,
                 input_cols: List[str] = None,
                 output_cols: List[str] = None,
                 list_group_by_cols: List[List[str]] = None):
        HasInputOutputCols.__init__(self, input_cols=input_cols, output_cols=output_cols)
        HasListGroupByCols.__init__(self, list_group_by_cols=list_group_by_cols)

    def impute(self, df: DataFrame) -> DataFrame:
        if len(self.input_cols) != len(self.output_cols):
            raise ValueError("Length of input columns must match length of output columns.")

        if self.list_group_by_cols is None:
            for i in range(len(self.input_cols)):
                in_col, out_col = self.input_cols[i], self.output_cols[i]

                df_mode = df \
                    .where(df[in_col].isNotNull()) \
                    .dropna(subset=in_col)

                counts = df_mode \
                    .groupBy(in_col) \
                    .count() \
                    .orderBy(["count", in_col], ascending=False)

                mode = counts.take(1)[0][0]

                df = df \
                    .withColumn(out_col, F.col(in_col)) \
                    .na.fill(mode, subset=out_col)

        else:

            if len(self.list_group_by_cols) != len(self.input_cols):
                raise ValueError("Length of List group by cols must match length of input columns.")

            for index in range(len(self.input_cols)):
                in_col, out_col, group_by_cols = self.input_cols[index], self.output_cols[index], \
                                                 self.list_group_by_cols[index]

                # warn_if_cols_have_nulls(df, group_by_cols)

                df_mode = df \
                    .where(df[in_col].isNotNull()) \
                    .dropna(subset=in_col)

                group_by_cols_and_in_col = group_by_cols[::]  # deep copy
                group_by_cols_and_in_col.append(in_col)
                tmp_col = in_col + "_tmp"   # needed so that we don't have issue with two `in_col` columns when joining.
                w = Window.partitionBy(group_by_cols)

                # calculates mode for each subgroup
                group_by_counts = df_mode \
                    .groupBy(group_by_cols_and_in_col) \
                    .count() \
                    .dropna(subset=in_col) \
                    .withColumn("max_count", F.max("count").over(w)) \
                    .where(F.col("count") == F.col("max_count")) \
                    .withColumnRenamed(in_col, tmp_col) \
                    .drop("count", "max_count")

                df_with_modes = df.join(group_by_counts, on=group_by_cols)

                df = df_with_modes \
                    .withColumn(out_col, F.coalesce(in_col, tmp_col)) \
                    .drop(tmp_col)

        return df
