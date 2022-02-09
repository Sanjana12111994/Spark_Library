#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:17:32 2020

@author: Kevin Shuai Zhang
"""

from pyspark.sql.functions import rand, randn
from pyspark.sql import SparkSession, Window
from pyspark.sql import SQLContext
import math
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Column
import sys
import unittest
from SparkSamplers import randnormal, randstudent, randchisq, randexp
    
def KolmogorovSmirnov2Test(df: DataFrame, column1: str, column2: str, significance = 0.05) -> bool:
    """
    Returns a True if two sample originate from the same population, or False if they are not. 
    
    References
    ----------
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    
    Parameters
    ----------
    df : pyspark dataframe
        A dataframe for hypothesis testing.
    column1 : str
        The name of first column of data.
    column2 : str
        The name of second column of data.
    significance: float
        Significance level of the test.  
    Returns
    -------
    result : bool 
        Result of the two-sampled, two-sided Kolmogorov–Smirnov test.
    """
    
    if column1 not in df.schema.names:
        raise TypeError(column1 + " is not in dataframe.")
    if column2 not in df.schema.names:
        raise TypeError(column2 + " is not in dataframe.")
        
    c1, n1 = ECDF(df, column1)
    c1 = c1.withColumnRenamed("cumulative", "cumulative1")
    c2, n2 = ECDF(df, column2)
    c2 = c2.withColumnRenamed("cumulative", "cumulative2")
    KS = KSStat2(c1, c2)
    T = KSTestStat2(n1, n2, significance)
    result = KS <= T
    return result   

def ECDF(df: DataFrame, col: str) -> (DataFrame, int):
    """
    Return an empirical CDFs constructed from the chosen columns from a pyspark dataframe.
    
    References
    ----------
    https://en.wikipedia.org/wiki/Empirical_distribution_function
    
    Parameters
    ----------
    df : pyspark dataframe
        An imputed dataframe.
    col : string
        The name of the chosen column of the dataframe to calculate empirical CDF.
    
    Returns
    -------
    c : pyspark dataframe 
        Empirical CDF caculated from the chosen column of the dataframe.
    n : int
        Number of rows in the chosen column of the dataframe.
    
    Examples
    --------
    >>> x = "normal"
    >>> df = df.select("id", randnormal(0, 1, seed = 10).alias(x))
    >>> htest = HypothesisTest()
    >>> c, n = htest.ECDF(df, x)
    >>> c.show()
    +--------------------+----------+
    |                   X|cumulative|
    +--------------------+----------+
    |  -1.866319666602813|       0.2|
    | -1.2370553024281632|       0.4|
    | -0.6471976318591632|       0.6|
    |-0.02656790187341...|       0.8|
    |-0.01560854938969...|       1.0|
    +--------------------+----------+
    """
    
    # count the number of not null        
    n = df.select(col).where(df[col].isNotNull()).count()
    
    f = df.groupBy(col).count()
    window = Window.orderBy(col).rangeBetween(Window.unboundedPreceding, 0)
    f = f.withColumn("cumulative", F.sum("count").over(window))
    
    c = f.select(df[col].alias("X"), (f["cumulative"] / n).alias("cumulative"))
    return c, n
    
def KSStat2(c1: str, c2: str) -> float:
    """
    Returns the two-sampled, two-sided Kolmogorov–Smirnov statistic, calculated from the empirical CDFs.
    
    Parameters
    ----------
    c1 : pyspark dataframe 
        First empirical CDF.
    c2 : pyspark dataframe 
        Second mpirical CDF.
    
    Returns
    -------
    D : float
        The two-sampled, two-sided Kolmogorov–Smirnov statistic, calculated from the empirical CDFs.
    
    Examples
    --------
    >>> htest = HypothesisTest()
    >>> D = htest.KSStat2(c1, c2)
    >>> print(D)
        0.4
    """
    
    x_id = c1.schema.names[0]
    df = c1.join(c2, on = [x_id], how = 'outer').sort(x_id, ascending=True)
    cols = ["cumulative1","cumulative2"]

    #Forward fill the ECDF to get a ECDF support.
    window_f = Window.rowsBetween(-sys.maxsize, 0)           
    df = df.replace(float('nan'), None)
    for col in cols:
        f_filled_column = F.last(df[col], ignorenulls=True).over(window_f)
        df_f_filled = df.withColumnRenamed(col, 'raw_' + col)
        df_f_filled = df.withColumn(col, f_filled_column)
        df = df_f_filled
        df = df.drop('raw_' + col)
        
    df = df.fillna(0, subset=col)
    df = df.select((df["cumulative1"] - df["cumulative2"]).alias("diff"))
    
    df = df.select(F.max(F.abs(df["diff"])).alias("max"), F.min(F.abs(df["diff"])).alias("min"))
    D = df.collect()[0][0]
    return D
    
def KSTestStat2(n1: int, n2: int, alpha: float) -> float:
    """
    Returns the test statistic of the two-sampled, two-sided Kolmogorov–Smirnov test.
    
    Parameters
    ----------
    n1 : int
        Size of the first sample.
    n2 : int
        Size of the second sample.
    alpha : float
        Significance level of the test.
    
    Returns
    -------
    T : float
        Test statistic of the two-sampled, two-sided Kolmogorov–Smirnov test.
    
    Examples
    --------
    >>> htest = HypothesisTest()
    >>> T = htest.TestStat(n1, n2)
    >>> print(T)
        0.8589388166934752
    """
        
    c = math.sqrt(-math.log(alpha/2) * 0.5)
    T = c * math.sqrt((n1+n2)/(n1*n2))
    return T

class _unittest(unittest.TestCase): 
    def test_one(self):
        df, x1, x2 = self.create_subject()
        df1 = df.select("id", randstudent(20, seed = 1).alias(x1), randnormal(0, 1, seed = 2).alias(x2))
        result = KolmogorovSmirnov2Test(df1, x1, x2)
        self.assertEqual(result, False)
    
    def test_two(self):
        df,x1,x2 = self.create_subject()
        df2 = df.select("id", randnormal(0, 1, seed = 3).alias(x1), randn(seed = 4).alias(x2))
        result = KolmogorovSmirnov2Test(df2, x1, x2)
        self.assertEqual(result, True)

    def test_three(self):
        df,x1,x2 = self.create_subject()
        df3 = df.select("id", randnormal(0, 1, seed = 5).alias(x1), randnormal(0, 2, seed = 6).alias(x2))
        result = KolmogorovSmirnov2Test(df3, x1, x2)
        self.assertEqual(result, False)
        
    def test_four(self):
        df,x1,x2 = self.create_subject()
        df3 = df.select("id", rand(seed = 7).alias(x1), rand(seed = 8).alias(x2))
        result = KolmogorovSmirnov2Test(df3, x1, x2)
        self.assertEqual(result, True)
    
    def test_five(self):
        df,x1,x2 = self.create_subject()
        df3 = df.select("id", randchisq(10, seed = 9).alias(x1), randchisq(20, seed = 10).alias(x2))
        result = KolmogorovSmirnov2Test(df3, x1, x2)
        self.assertEqual(result, False)
    
    def test_six(self):
        df,x1,x2 = self.create_subject()
        df3 = df.select("id", randexp(5, seed = 11).alias(x1), randexp(5, seed = 12).alias(x2))
        result = KolmogorovSmirnov2Test(df3, x1, x2)
        self.assertEqual(result, True)
    
    def create_subject(self):
        spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()   
        sqlContext = SQLContext(spark)
        # Adjust the number of rows here
        size = 100000
        df = sqlContext.range(0, size)
        x1 = "X1"
        x2 = "X2"
        return df, x1, x2

        

if __name__ == '__main__':

    unittest.main()
