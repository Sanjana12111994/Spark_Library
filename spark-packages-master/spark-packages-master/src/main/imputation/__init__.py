import sys, os
from pathlib import Path
#sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../main")
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
from typing import List, Dict
from pyspark.sql import Window
from pyspark.sql.functions import last, first  
import os
from pathlib import Path
path = Path(__file__).resolve().parents[2]
os.chdir(path)

#from smartImputations import imputation
def main():
    # Spark Session
    spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    sc = spark.sparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    # Load Dataset
    path = Path(__file__).resolve().parents[3]
    file = str(path) + "/datasets/missing.csv"
    print(file)
    raw_df = sqlContext.read.load(file,
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')
    raw_df = raw_df.withColumnRenamed("_c0","id")
    raw_df.show()
    #Tests b_fill
    #backimp = imputation(strategy = "bfill", output_cols = ["age","age1"])
    #df = backimp.impute(raw_df)
    #df.show()
    
    # Tests f_fill
    #forwadimp = imputation(strategy = "ffill", output_cols = ["age","age1"])
    #df = forwadimp.impute(raw_df)
    #df.show()
        
    # Tests 1
# =============================================================================
#     linimp = imputation(strategy = "linreg", input_cols = ["income","gender"], output_cols = ["age"])
#     df = linimp.impute(raw_df)
#     df.show()
# =============================================================================
    #Test MICE 
    miceimp = SmartImputer(strategy = "mice", output_cols = ["age","income","gender"])
    df = miceimp.impute(raw_df)
    df.show()
    # Tests 2
    #mlinimp = imputation(strategy = "mlinreg", input_cols = ["crim"], output_cols = ["zn", "indus"])
    #df = mlinimp.impute(raw_df)
    #df.show()
if __name__ == "__main__":
    main()
