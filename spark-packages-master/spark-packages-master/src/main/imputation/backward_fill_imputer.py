import sys
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window

from .imputer_params import HasInputOutputCols, HasOrderByCols, HasListPartitionByCols


class BackwardFillImputer(HasInputOutputCols, HasOrderByCols, HasListPartitionByCols):
    """
    An imputation class to impute by backward fill for a single column, or by partitions.

    Parameters
    ----------
    input_cols : List[str]
        A list of column names with missing values to be imputed.
    output_cols : List[str]
        A list of column names where for index i in 0, ..., n-1 for
        input_cols of length n, input_cols[i] -> output_cols[i].
        Defaults to equaling input_cols and overwrites the input columns.
    order_by_cols : List[str]
        The column to give meaningful order to the DataFrame when using
        a Window function.
    list_partition_by_cols : List[List[str]]
        A List of List of column names at index i in 0, ..., n-1
        to do imputation by partitions on. Must be same length as `input_cols`.

    See Also
    --------
    HasInputOutputCols, HasOrderByCols, HasListPartitionByCols

    Notes
    -----
    Single imputation methods treat the imputed value as the "true" value, which does not account for any
    uncertainty in the imputation, often resulting in variance being underestimated as well as an
    overestimation of precision.

    References
    ----------
    [1] Donders et al. (2006). Review: A gentle introduction to imputation of missing values.
        Journal of Clinical Epidemiology, 59(10), 1087-1091.

    Examples
    --------
    >>> schema = StructType(\
        [StructField("CustomerID", IntegerType()),\
         StructField("Date", StringType()),\
         StructField("Balance", IntegerType())])
    >>> df = spark.createDataFrame([(10001, "2020-01-01", 145), (10002, "2020-01-01", 474),\
                                    (10001, "2020-01-02", None), (10002, "2020-01-02", None), \
                                    (10001, "2020-01-03", 155), (10002, "2020-01-03", 356)], schema=schema) \
                                    .withColumn("Date", F.to_date(F.col("Date"), "yyyy-mm-dd"))
    >>> bfill_imputer = BackwardFillImputer(input_cols=["Balance"], output_cols=["BalanceImp"], order_by_cols=["Date"])
    >>> bfill_imputer.impute(df).show()
    +----------+----------+-------+----------+
    |CustomerID|      Date|Balance|BalanceImp|
    +----------+----------+-------+----------+
    |     10001|2020-01-01|    145|       145|
    |     10002|2020-01-01|    474|       474|
    |     10001|2020-01-02|   null|       155|
    |     10002|2020-01-02|   null|       155|
    |     10001|2020-01-03|    155|       155|
    |     10002|2020-01-03|    356|       356|
    +----------+----------+-------+----------+

    >>> bfill_imputer.list_partition_by_cols = [["CustomerId"]]
    >>> bfill_imputer.impute(df).orderBy("Date", "CustomerId").show()
    +----------+----------+-------+----------+
    |CustomerID|      Date|Balance|BalanceImp|
    +----------+----------+-------+----------+
    |     10001|2020-01-01|    145|       145|
    |     10002|2020-01-01|    474|       474|
    |     10001|2020-01-02|   null|       155|
    |     10002|2020-01-02|   null|       356|
    |     10001|2020-01-03|    155|       155|
    |     10002|2020-01-03|    356|       356|
    +----------+----------+-------+----------+

    """
    def __init__(self,
                 input_cols: List[str] = None,
                 output_cols: List[str] = None,
                 order_by_cols: List[str] = None,
                 list_partition_by_cols: List[List[str]] = None):

        HasInputOutputCols.__init__(self, input_cols=input_cols, output_cols=output_cols)
        HasOrderByCols.__init__(self, order_by_cols=order_by_cols)
        HasListPartitionByCols.__init__(self, list_partition_by_cols=list_partition_by_cols)

    def impute(self, df: DataFrame) -> DataFrame:
        if len(self.input_cols) != len(self.output_cols):
            raise ValueError("Length of input columns must match length of output columns.")

        # check if lengths match up
        if self.list_partition_by_cols is not None:
            if len(self.list_partition_by_cols) != len(self.input_cols):
                raise ValueError("Length of List partition by cols must match length of input columns.")

        for i in range(len(self.input_cols)):
            in_col, out_col, order_by_col = self.input_cols[i], self.output_cols[i], self.order_by_cols[i]

            # Set the window dimension
            if self.list_partition_by_cols is None:
                w = Window \
                    .orderBy(order_by_col) \
                    .rowsBetween(0, sys.maxsize)
            else:
                partition_by_cols = self.list_partition_by_cols[i]
                w = Window \
                    .partitionBy(partition_by_cols) \
                    .orderBy(order_by_col) \
                    .rowsBetween(0, sys.maxsize)

            # define the backward-filled column
            b_filled_column = F.first(df[in_col], ignorenulls=True).over(w)

            df = df.withColumn(out_col, b_filled_column)

        return df

