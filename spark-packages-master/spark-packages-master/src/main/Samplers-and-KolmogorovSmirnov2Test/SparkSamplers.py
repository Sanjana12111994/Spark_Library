#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains various probabilstic sampling methods as well as the Reservoir Sampling method

    Probabilstic methods: General Normal, Central Chi-Square, Central Student-T, Central Exponential
    
    Reservoir Sampling algorithm: Algorithm L by  Li, Kim-Hung

Created on Fri Aug 12 09:09:18 2020

@author: Kevin Shuai Zhang
"""

from pyspark.sql.functions import rand, randn, log, exp, when
from pyspark.sql import SparkSession, Window
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
import unittest

def randnormal(mean: float, std: float, seed = None):
    """
    Generates a column with independent and identically distributed (i.i.d.) samples from
    a general normal distribution.

    .. note:: The function is non-deterministic in general case.
    
    Parameters
    ----------
    mean : float
        The mean of normal distribution.
        
    std : float
        The standard deviation of normal distribution.
        
    Seed : int
        A random seed.
    
    Returns
    -------
    col : pyspark.sql.column Column 
        A pyspark DataFrame column of i.id. normal random variables.
        
    Example
    -------
    >>> df = sqlContext.range(0, 5)
    >>> df = df.withColumn("normal", randnormal(0, 1, seed = 3))
    >>> df.show()
        +---+--------------------+
        | id|              normal|
        +---+--------------------+
        |  0| -1.1081822375859998|
        |  1|-0.24587658470592705|
        |  2|  -1.773755556110447|
        |  3|  1.8039683668407596|
        |  4| -0.4823162289462346|
        +---+--------------------+
    """
    
    if seed is not None:
        col = randn(seed) * std + mean
    else:
        col = randn() * std + mean
    return col

def randchisq(degree: int, seed = None):
    """
    Generates a column with independent and identically distributed (i.i.d.) samples from
    a centralized Chi-Square distribution.

    .. note:: The function is non-deterministic in general case.
    
    References
    ----------
    https://en.wikipedia.org/wiki/Chi-squared_test
    
    Parameters
    ----------
    degree : int
        The degree of freedom of chi-square distribution.
    Seed : int
        A random seed.
    
    Returns
    -------
    col : pyspark.sql.column Column 
        A pyspark dataframe column of i.id. Chi-Square random variables.
    
    Example
    -------
    >>> df = sqlContext.range(0, 5)
    >>> df = df.withColumn("chisq", randchisq(2, seed = 3))
    >>> df.show()
        +---+-------------------+
        | id|              chisq|
        +---+-------------------+
        |  0| 2.4561357434022266|
        |  1|0.12091058981330184|
        |  2|  6.292417545665362|
        |  3| 6.5086037371242345|
        |  4| 0.4652578894098332|
        +---+-------------------+
    """
    
    if seed is not None:
        col = pow(randn(seed = seed), 2)
        for i in range(degree - 1):
            col = col + pow(randn(seed = seed), 2)
    else:
        col = pow(randn(), 2)
        for i in range(degree - 1):
            col = col + pow(randn(), 2)
    return col

def randstudent(degree, seed = None):
    """
    Generates a column with independent and identically distributed (i.i.d.) samples from
    a centralized Student-T distribution.

    .. note:: The function is non-deterministic in general case.
    
    References
    ----------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    
    Parameters
    ----------
    degree : int
        The degree of freedom of Student-T distribution.
    Seed : int
        A random seed.
    
    Returns
    -------
    col : pyspark.sql.column Column 
        A pyspark dataframe column of i.id. Student-T random variables.
    
    Example
    -------
    >>> df = sqlContext.range(0, 5)
    >>> df = df.withColumn("Student-T", randstudent(2, seed = 3))
    >>> df.show()
        +---+--------------------+
        | id|           Student-T|
        +---+--------------------+
        |  0| -0.6247660416162757|
        |  1|-0.13629761431821777|
        |  2| -3.6775780072458923|
        |  3|   9.508153844519372|
        |  4| -18.154095541461576|
        +---+--------------------+
    """
    
    if seed is not None:
        col = randn(seed = seed) / pow((randchisq(degree, seed+seed) / degree), 0.5)
    else:
        col = randn() / pow((randchisq(degree) / degree), 0.5)
    return col

def randexp(rate: float, seed = None):
    """
    Generates a column with independent and identically distributed (i.i.d.) samples from
    a 1 parameter exponential distribution.

    .. note:: The function is non-deterministic in general case.
    
    References
    ----------
    https://en.wikipedia.org/wiki/Exponential_distribution
    
    Parameters
    ----------
    rate : float
        The scale of exponential distribution.
        
    Seed : int
        A random seed.
    
    Returns
    -------
    col : pyspark.sql.column Column 
        A pyspark DataFrame column of i.id. exponential random variables.
        
    Example
    -------
    >>> df = sqlContext.range(0, 5)
    >>> df = df.withColumn("exponential", randexp(0, 1, seed = 3))
    >>> df.show()
        +---+-------------------+
        | id|        exponential|
        +---+-------------------+
        |  0|  1.530352502988964|
        |  1|0.03649956819614562|
        |  2|0.16556014234412864|
        |  3| 0.6268740044366697|
        |  4|0.09373723340752678|
        +---+-------------------+
    """
    
    if seed is not None:
        col = - log(1 - rand(seed)) / rate
    else:
        col = - log(1 - rand()) / rate
    return col

def ReservoirSampler(df: DataFrame, sample_size: int, 
                     population_size = None, 
                     columns = None, 
                     seed = None) -> DataFrame:
    
    """
    Pyspark implementation of Reservoir sampling 

    .. note:: This function uses Algorithm L
    
    References
    ----------
    https://en.wikipedia.org/wiki/Reservoir_sampling
    
    Parameters
    ----------
    df : pyspark DataFrame
        The DataFrame to sample from.    
    sample_size : int
        Size of the sample reservoir.
    population_size : int
        Size of the population.
    columns : list
        A list of columns names to sample from.
    seed : int
        Seed of the sampling.
    
    Returns
    -------
    reservoir_df : pyspark DataFrame 
        A pyspark DataFrame contains the reservoir sample.
        
    Example
    -------
    >>> df.show()
        +---+--------------------+--------------------+
        | id|                  U1|                  U2|
        +---+--------------------+--------------------+
        |  1| 0.14878636881596974|  0.3787049738242783|
        |  2|   1.530352502988964| 0.14878636881596974|
        |  3|0.012098686811669868|   1.530352502988964|
        |  4| 0.03649956819614562|0.012098686811669868|
        |  5|  0.6662892395896088| 0.04996413599051409|
        |  6| 0.16556014234412864| 0.03649956819614562|
        |  7|   0.330486051772355| 0.16556014234412864|
        |  8|  0.6268740044366697|   0.330486051772355|
        |  9| 0.09373723340752678|  0.6268740044366697|
        | 10|  0.8176712688464889| 0.31388364689779324|
        +---+--------------------+--------------------+
    >>> RDF = ReservoirSampler(df, sample_size=5, seed=1)
    >>> RDF.show()
        +---+--------------------+-------------------+
        | id|                  U1|                 U2|
        +---+--------------------+-------------------+
        |  1| 0.14878636881596974| 0.3787049738242783|
        |  2| 0.16556014234412864|0.03649956819614562|
        |  3|0.012098686811669868|  1.530352502988964|
        |  4|  0.8176712688464889|0.31388364689779324|
        |  5|  0.6268740044366697|  0.330486051772355|
        +---+--------------------+-------------------+
    
    """
    
    # Initialize seed
    if seed is not None:
        seed_1 = seed
        seed_2 = 2 * seed
        seed_3 = 3 * seed
    else:
        seed_1 = None
        seed_2 = None
        seed_3 = None 
        
    # Initialize sampling columns
    if columns is None:
        columns = df.schema.names
        columns.remove('id')
 
    # Initialize population size
    if population_size is None:
        population_size = df.count()
        
    free_range_size = population_size - sample_size
        
    # Initialize the reservoir sample df
    reservoir_df = df.limit(sample_size)
    
    # Initialize the replacement mapping df
    map_df = df.select("id").limit(free_range_size)
    
    # Generate a Markov Chain of replacement indices from Geometric Distribution
    window = Window.orderBy("id").rangeBetween(Window.unboundedPreceding, 0)
    W_col = exp(F.sum(log(rand(seed_1)) / sample_size).over(window))
    S_col = F.sum(F.floor(log(rand(seed_2)) / log(1 - W_col)) + 1).over(window) + sample_size
    R_col = F.round(rand(seed_3) * (sample_size - 1) + 1, 0).cast(IntegerType())
    
    # Create the replacement mapping df
    map_df = map_df.select("id", S_col.alias("S"), R_col.alias("R")) \
            .orderBy("S", ascending=False)
            
    map_df = map_df.where(map_df["S"] <= population_size) \
            .dropDuplicates(["R"])
    
    # Replace the resevoir sample according to the replacement mapping
    dummy_df = map_df.select("S","R", map_df["S"].alias("id"))
    dummy_df = df.join(dummy_df, on = ["id"], how = 'outer')
    dummy_df = dummy_df.filter(dummy_df["S"].isNotNull())
    dummmy_col = []
    for col in columns:
        dummy_df = dummy_df.withColumnRenamed(col, "d"+col)
        dummmy_col.append("d"+col)
    dummmy_col.append("R")
    dummy_df = dummy_df.select(dummmy_col)
    dummy_df = dummy_df.withColumnRenamed("R", "id")
    reservoir_df = reservoir_df.join(dummy_df, on = ["id"], how = 'outer')
    for col in columns:
        reservoir_df = reservoir_df.withColumn(col, when(reservoir_df["d"+col] \
                        .isNotNull(), reservoir_df["d"+col]).otherwise(reservoir_df[col]))
    columns = ["id"] + columns
    reservoir_df= reservoir_df.select(columns)

    return reservoir_df

class _unittest(unittest.TestCase): 
    def test_one(self):
        spark = self.create_spark()
        sqlContext = SQLContext(spark)
        expected_df = spark.createDataFrame([(1, 0.14878636881596974), 
                                                  (2, 0.330486051772355), 
                                                  (3, 0.012098686811669868), 
                                                  (4, 0.6268740044366697), 
                                                  (5, 0.09373723340752678)], 
                                                 ("id", "E1"))
        df = sqlContext.range(1, 10)
        df = df.withColumn("U1", randexp(2, seed = 3))
        RDF = ReservoirSampler(df, sample_size = 5, seed=1)
        result = self.compare_df(RDF, expected_df)
        self.assertEqual(result, True)
        
    def test_two(self):
        spark = self.create_spark()
        sqlContext = SQLContext(spark)
        expected_df = spark.createDataFrame([(0, 1.530352502988964), 
                                                  (1, 0.03649956819614562), 
                                                  (2, 0.16556014234412864), 
                                                  (3, 0.6268740044366697), 
                                                  (4, 0.09373723340752678)], 
                                                 ("id", "E1"))
        df = sqlContext.range(0, 5)
        df = df.withColumn("U1", randexp(2, seed = 3))
        result = self.compare_df(df, expected_df)
        self.assertEqual(result, True)
        
    def test_three(self):
        spark = self.create_spark()
        sqlContext = SQLContext(spark)
        expected_df = spark.createDataFrame([(0, -1.1081822375859998), 
                                                  (1, -0.24587658470592705), 
                                                  (2, -1.773755556110447), 
                                                  (3, 1.8039683668407596), 
                                                  (4, -0.4823162289462346)], 
                                                 ("id", "N1"))
        df = sqlContext.range(0, 5)
        df = df.withColumn("N1", randnormal(0, 1, seed = 3))
        result = self.compare_df(df, expected_df)
        self.assertEqual(result, True)
        
    def test_four(self):
        spark = self.create_spark()
        sqlContext = SQLContext(spark)
        expected_df = spark.createDataFrame([(0, 6.140339358505567), 
                                                  (1, 0.3022764745332546), 
                                                  (2, 15.731043864163405), 
                                                  (3, 16.271509342810585), 
                                                  (4, 1.163144723524583)], 
                                                 ("id", "C1"))
        df = sqlContext.range(0, 5)
        df = df.withColumn("C1", randchisq(5, seed = 3))
        result = self.compare_df(df, expected_df)
        self.assertEqual(result, True)
        
    def test_five(self):
        spark = self.create_spark()
        sqlContext = SQLContext(spark)
        expected_df = spark.createDataFrame([(0, -0.6247660416162757), 
                                                  (1, -0.13629761431821777), 
                                                  (2, -3.677578007245892), 
                                                  (3, 9.508153844519372), 
                                                  (4, -18.154095541461576)], 
                                                 ("id", "T1"))
        df = sqlContext.range(0, 5)
        df = df.withColumn("T1", randstudent(5, seed = 3))
        result = self.compare_df(df, expected_df)
        self.assertEqual(result, True)
        
    def compare_df(self, df1, df2):
        return df1.collect() == df2.collect()
        
    def create_spark(self):
        spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()         
        return spark
        
if __name__ == '__main__':    
    unittest.main()
    