import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../main")
from pathlib import Path
from pyspark.sql.types import StructType
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from pathlib import Path
from FeatureEncoder import FeatureEncoder

path = Path(__file__).resolve().parents[0]
os.chdir(path)
from FeatureEncoder import FeatureEncoder

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
#    file = str(path) + "/datasets/missing.csv"
#    print(file)
    #Create dataset
    schema = StructType([StructField("Country", StringType(),True), 
                         StructField("Label", StringType(),True)])
                         
    data = [("US","NO"),("CA","YES"),("IN","NO"),("IN","YES"),("US","NO"),("CA","YES"),("IN","NO"),("CA","YES"),("CA","NO"),("CA","YES"),("US","NO"),("CA","YES")]
    
    raw_df = spark.createDataFrame(data, schema=schema)
    raw_df.show()
    print(raw_df)
    #Test Supervised Ratio 
    encoder = FeatureEncoder(strategy = "WOE",cat_col="Country",label_col="Label",positive_class="YES",negative_class="NO")
    df = encoder.encode(raw_df)
    df.show()
  
if __name__ == "__main__":
    main()
