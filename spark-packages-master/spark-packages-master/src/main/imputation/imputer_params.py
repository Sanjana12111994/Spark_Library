from typing import List


class HasInputOutputCols:
    """
    Parameters
    ----------
    input_cols : List[str]
        A list of column names with missing values to be imputed.
    output_cols : List[str]
        A list of column names where for index i in 0, ..., n-1 for
        input_cols of length n, input_cols[i] -> output_cols[i].
        Defaults to equaling input_cols and overwrites the input columns.

    Notes
    -----
    Input and Output columns cannot be decoupled since `output_cols` are dependent on
    the value of `input_cols` when it is not specified.
    """
    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None):
        self.input_cols = input_cols
        self.output_cols = output_cols

    @property
    def input_cols(self):
        return self._input_cols

    @input_cols.setter
    def input_cols(self, value: List[str]):
        if not isinstance(value, list):
            raise TypeError("Input columns must be in a list.")
        elif len(value) == 0:
            raise ValueError("Input columns cannot be length 0.")
        for col in value:
            if not isinstance(col, str):
                raise TypeError("Input column names must be strings.")
            elif len(col) == 0:
                raise ValueError("Input columns cannot be empty string.")
        self._input_cols = value

    @property
    def output_cols(self):
        return self._output_cols

    @output_cols.setter
    def output_cols(self, value: List[str]):
        if value is None:
            self._output_cols = self.input_cols
            return  # exit early if not output_cols not specified and default to overwriting input_cols.
        if not isinstance(value, list):
            raise TypeError("Output columns must be in a list.")
        elif len(value) == 0:
            raise ValueError("Output columns cannot be length 0.")
        for col in value:
            if not isinstance(col, str):
                raise TypeError("Output column names must be strings.")
            elif len(col) == 0:
                raise ValueError("Output columns cannot be empty string.")
        self._output_cols = value


class HasMissingValue:
    """
    Parameters
    ----------
    # TODO: rework other imputation methods besides Mean/Median so that they can use this parameter.
    missing_value : float
        A specified value to be imputed in addition to `null`.

    Notes
    -----
    Only usable with `MeanImpute` and `MedianImputer` as of 2020-08-10

    """
    def __init__(self, missing_value: float = float("nan")):
        self.missing_value = missing_value

    @property
    def missing_value(self):
        return self._missing_value

    @missing_value.setter
    def missing_value(self, value):
        if value is None:
            raise ValueError("Missing value cannot be None.")
        self._missing_value = value


class HasRelativeError:
    """
    Parameters
    ----------
    relative_error : float
        The margin of error allowed for imputing by the median.
        Defaults to 0.001 as per pyspark.ml.feature.Imputer.

    Notes
    -----
    Only used with `MedianImputer`

    """
    def __init__(self, relative_error: float = 0.001):
        self.relative_error = relative_error

    @property
    def relative_error(self):
        return self._relative_error

    @relative_error.setter
    def relative_error(self, value):
        if value is None:
            raise ValueError("Relative error cannot be None.")
        self._relative_error = value


class HasOrderByCols:
    """
    Parameters
    ----------
    order_by_cols : List[str]
        The column to give meaningful order to the DataFrame when using
        a Window function.

    Notes
    -----
    Only used by `ForwardFillImputer` and `BackwardFillImputer`.

    """
    def __init__(self, order_by_cols: List[str] = None):
        self.order_by_cols = order_by_cols

    @property
    def order_by_cols(self):
        return self._order_by_cols

    @order_by_cols.setter
    def order_by_cols(self, value: List[str] = None):
        if not isinstance(value, list):
            raise TypeError("Order by columns must be in a list.")
        elif len(value) == 0:
            raise ValueError("Order by columns cannot be length 0.")
        for col in value:
            if not isinstance(col, str):
                raise TypeError("Order by column names must be strings.")
            elif len(col) == 0:
                raise ValueError("Order by columns cannot be empty string.")

        self._order_by_cols = value


class HasListPartitionByCols:
    """
    Parameters
    ----------
    list_partition_by_cols : List[List[str]]
        A List of List of column names at index i in 0, ..., n-1
        to do imputation by partitions on. Must be same length as `input_cols`.

    Notes
    -----
    Only used by `ForwardFillImputer` and `BackwardFillImputer`.

    """
    def __init__(self, list_partition_by_cols: List[List[str]] = None):
        self.list_partition_by_cols = list_partition_by_cols

    @property
    def list_partition_by_cols(self):
        return self._list_partition_by_cols

    @list_partition_by_cols.setter
    def list_partition_by_cols(self, value: List[List[str]]):
        if value is None:
            self._list_partition_by_cols = value
            return  # break out early since this parameter is optional
        elif not isinstance(value, list):
            raise TypeError("List partition by cols must be a List of List of strings.")

        for partition_by_cols in value:
            if not isinstance(partition_by_cols, list):
                raise TypeError("Each partition by cols must be a List of strings.")
            # TODO: consider adding check for len(partition_by_cols) == 0.
            #       Might make sense to allow though, since it could indicate *not* doing a group_by on that input_col.
            for col in partition_by_cols:
                if not isinstance(col, str):
                    raise TypeError("Columns must be a string.")
                elif len(col) == 0:
                    raise ValueError("Column names cannot be empty string.")
        self._list_partition_by_cols = value


class HasListGroupByCols:
    """
    Parameters
    ----------
    list_group_by_cols : List[List[str]]
        A List of List of column names at index i in 0, ..., n-1
        to do imputation by subgroups on. Must be same length as `input_cols`.

    Notes
    -----
    Used by `MeanImputer`, `MedianImputer`, `ModeImputer`.

    Can be extended to be used by any single imputation method
    that makes sense to perform on subgroups.

    """
    def __init__(self, list_group_by_cols: List[List[str]] = None):
        self.list_group_by_cols = list_group_by_cols

    @property
    def list_group_by_cols(self):
        return self._list_group_by_cols

    @list_group_by_cols.setter
    def list_group_by_cols(self, value: List[List[str]]):
        if value is None:
            self._list_group_by_cols = value
            return  # break out early since this parameter is optional
        elif not isinstance(value, list):
            raise TypeError("List group by cols must be a List of List of strings.")

        for group_by_cols in value:
            if not isinstance(group_by_cols, list):
                raise TypeError("Each group by cols must be a List of strings.")
            # TODO: consider adding check for len(group_by_cols) == 0.
            #       Might make sense to allow though, since it could indicate *not* doing a group_by on that input_col.
            for col in group_by_cols:
                if not isinstance(col, str):
                    raise TypeError("Columns must be a string.")
                elif len(col) == 0:
                    raise ValueError("Column names cannot be empty string.")

        self._list_group_by_cols = value


class HasStrategy:
    """
    Parameters
    ----------
    strategy : str
        The single imputation method to use in its most simple form.
        Defaults to `mean` if not specified.
        Currently, `mean`, `median`, `mode`, `ffill`, and `bfill` are supported.
        Any imputation method that can map `input_cols`[i] -> `output_cols`[i] can be implemented.

    Notes
    -----
    Only used by `SimpleImputer`.

    """
    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value: str):
        valid_strategies = ["mean", "median", "mode", "ffill", "bfill"]

        if not isinstance(value, str):
            raise TypeError("Strategy must be a string.")
        elif value.lower() not in valid_strategies:
            raise ValueError("""Strategy: {} is not supported. Current valid strategies are:
                                    'mean', 'median', 'mode', 'ffill', 'bfill'."""
                             .format(value))
        self._strategy = value.lower()
