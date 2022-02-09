from typing import List

from pyspark.sql import DataFrame

from .imputer_params import HasInputOutputCols, HasMissingValue, HasRelativeError, HasOrderByCols, HasStrategy
from .mean_imputer import MeanImputer
from .median_imputer import MedianImputer
from .mode_imputer import ModeImputer
from .forward_fill_imputer import ForwardFillImputer
from .backward_fill_imputer import BackwardFillImputer


class SimpleImputer(HasStrategy, HasInputOutputCols, HasMissingValue, HasRelativeError, HasOrderByCols):
    """
    A class that acts as a wrapper for Mean, Median, Mode, Forward/Backward Fill imputation in their most simple form.

    For more nuanced usage of the methods, use the individual imputers themselves.

    Parameters
    ----------
    strategy : str
        The single imputation method to use in its most simple form.
        Defaults to `mean` if not specified.
        Currently, `mean`, `median`, `mode`, `ffill`, and `bfill` are supported.
        Any imputation method that can map `input_cols`[i] -> `output_cols`[i] can be implemented.
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
    order_by_cols : List[str]
        The column to give meaningful order to the DataFrame when using
        a Window function.

    See Also
    -------
    HasStrategy, HasInputOutputCols, HasMissingValue, HasRelativeError, HasOrderByCols

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
    -------
    >>> df = spark.createDataFrame([(1.0,), (2.0,), (3.0,), (float("nan"), )], ["values"])
    >>> simple_imputer = SimpleImputer(input_cols=["values"])
    >>> simple_imputer.impute(df).show()
    +------+
    |values|
    +------+
    |   1.0|
    |   2.0|
    |   3.0|
    |   2.0|
    +------+

    >>> simple_imputer.output_cols = ["values_imputed"]
    >>> simple_imputer.strategy = "median"
    >>> simple_imputer.impute(df).show()
    +------+--------------+
    |values|values_imputed|
    +------+--------------+
    |   1.0|           1.0|
    |   2.0|           2.0|
    |   3.0|           3.0|
    |   NaN|           2.0|
    +------+--------------+

    """
    def __init__(self,
                 strategy: str = "mean",
                 input_cols: List[str] = None,
                 output_cols: List[str] = None,
                 order_by_cols: List[str] = ["placeholder"],
                 missing_value: float = float("nan"),
                 relative_error: float = 0.001):

        HasInputOutputCols.__init__(self, input_cols=input_cols, output_cols=output_cols)
        HasMissingValue.__init__(self, missing_value=missing_value)
        HasRelativeError.__init__(self, relative_error=relative_error)
        HasOrderByCols.__init__(self, order_by_cols=order_by_cols)
        HasStrategy.__init__(self, strategy=strategy)

    def impute(self, df: DataFrame) -> DataFrame:
        """
        Perform the single imputation by chosen strategy.
        """
        strategy = self.strategy

        if strategy == "mean":
            imputer = MeanImputer(input_cols=self.input_cols,
                                  output_cols=self.output_cols,
                                  missing_value=self.missing_value)
        elif strategy == "median":
            imputer = MedianImputer(input_cols=self.input_cols,
                                    output_cols=self.output_cols,
                                    missing_value=self.missing_value,
                                    relative_error=self.relative_error)
        elif strategy == "mode":
            imputer = ModeImputer(input_cols=self.input_cols,
                                  output_cols=self.output_cols)
        elif strategy == "ffill":
            imputer = ForwardFillImputer(input_cols=self.input_cols,
                                         output_cols=self.output_cols,
                                         order_by_cols=self.order_by_cols)
        else:
            imputer = BackwardFillImputer(input_cols=self.input_cols,
                                          output_cols=self.output_cols,
                                          order_by_cols=self.order_by_cols)

        return imputer.impute(df=df)
