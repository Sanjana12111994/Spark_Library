import warnings
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


# Taken from: https://stackoverflow.com/questions/55964196/check-whether-dataframe-contains-any-null-values
def warn_if_cols_have_nulls(df: DataFrame, cols: List[str]):
    cols_with_nulls = []
    df_subset = df[cols]
    for c in df_subset.columns:
        cur_col = F.col(c)
        if not df_subset.where(cur_col.isNull() | F.isnan(cur_col)).limit(1).collect():
            cols_with_nulls.append(c)
    # non-empty list check
    if cols_with_nulls:
        cols_string = ", ".join(cols_with_nulls)
        message = f"""The following columns contain null values that will
                      be treated as groups during a GROUP BY: {cols_string}."""
        warnings.warn(message=message, category=UserWarning)


