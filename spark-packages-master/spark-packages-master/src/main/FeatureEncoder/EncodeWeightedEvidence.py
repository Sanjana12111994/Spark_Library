import sys
import random
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import log10


def encode_weighted_evidence(df: DataFrame,
                 cat_col: str,
                 label_col: str,
                 positive_class: str,
                 negative_class: str,
                 bias: bool
                ) -> DataFrame:


    """ A function to encode categorical features with Weight of Evidence Ratio
    General idea of code inspired by:
        https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html


        :param df               :  The dataframe which contains categorical features to be encoded.
        :param cat_col          :  The String that denotes the categorical column name.
        :param label_col        :  The String that denotes the label/target column name.
        :param positive_class   :  The String value in the label_col to denote positive class label. 
                                   default value = '1'
        :param negative_class   :  The String value in the label_col to denote negative class label.
                                   default value = '0'
        :bias                   :  A flag variable to indicate whether or not to include bias while encoding feature
                                   default value = False

        Eg:
        >>> schema = StructType([StructField("Country", StringType(),True), 
                         StructField("Label", IntegerType(),True)])                 
        >>> data = [("US",0),("CA",1),("IN",0),("IN",1),("US",0),("CA",1),("IN",0),("CA",1),("CA",0),("CA",1),("US",0),("CA",1)]
        >>> raw_df = spark.createDataFrame(data, schema=schema)
        >>> encoder = FeatureEncoder(strategy = "WOE",cat_col="Country",label_col="Label")
        >>> df = encoder.encode(raw_df)
    """

   
    #Count the number distinct category with their respective class
    df1 = df.groupby(cat_col,label_col).count() 
    #Count the number of records
    total_count = df.count()
    
    #Create two seperate dataframes for positive and negative class
    #Dataframe containing positive class
    df_class1 = df1.select(cat_col,label_col,'count').where(df[label_col]== positive_class)
    df_class1.cache()
    df_class1 = df_class1.withColumnRenamed("count","count_c1").dropDuplicates().drop(label_col)

    #Dataframe containing negative class
    df_class0 = df1.select(cat_col,label_col,'count').where(df[label_col]== negative_class)
    df_class0.cache()
    df_class0 = df_class0.withColumnRenamed("count","count_c0").dropDuplicates().drop(label_col)



    #Calculate the Weighted Average Encoding for all the cases 
    total_C0 = df_class0.select(F.sum("count_c0")).take(1)[0][0]
    total_C1 = df_class1.select(F.sum("count_c1")).take(1)[0][0]
    
    # Join both the dataframes positive class and negative class
    df_join = df_class0.join(df_class1, on = [cat_col], how = "full").na.fill(0)
    
    we_col = (df_join['count_c1'] / total_C1) / (df_join['count_c0'] / total_C0)
    df_join =df_join.withColumn("WOE",we_col).na.fill(0) 


    #Special Case Handling : 
    #where the value is either missing or when the number of positive or negative values for a category is zero
    if bias:
        df_join =df_join.withColumn("penalty",F.rand()*0.1 - 0.06) 
        
    df_join = df_join.withColumn("WOE_prime", \
                F.when((df_join["count_c1"] == 0) | (df_join["count_c0"] == 0) ,\
                        (F.col("count_c0") + F.col("count_c1")) / total_count)\
                                .otherwise(df_join["WOE"]))

   
    if bias:
        df_encoded = df.join(broadcast(df_join), on=[cat_col]).drop('count_c0','count_c1','WOE')
        df_encoded = df_encoded.withColumn(cat_col+"_encoded",log10("WOE_prime") + F.col("penalty")).drop("WOE_prime","penalty")
    else:
        df_encoded = df.join(broadcast(df_join), on=[cat_col]).drop('count_c0','count_c1','WOE')
        df_encoded = df_encoded.withColumn(cat_col+"_encoded",log10("WOE_prime")).drop("WOE_prime")
    
      
    return df_encoded