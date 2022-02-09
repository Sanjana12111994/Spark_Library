# Spark-Packages

## imputation
A package for filling in missing data through a variety of single imputation methods.


There are 6 imputation methods implemented so far: `mean`, `median`, `mode`,
`forward fill`, `backward fill`, `linear regression imputation`. 

In addition to those, 
there is `SimpleImputer` which is a flexible class that acts as a wrapper around 
the first 5  methods in their most simple forms. 

SimpleImputer should be used primarily
for exploratory data analysis (EDA) only, as most imputation cases will require more nuanced
usage in the form of using the supported `GROUP BY` and `PARTITION BY` parameters,
`list_group_by_cols` and `list_partition_by_cols`.

### Usage
#### SimpleImputer
```python
from src.main.imputation.simple_imputer import SimpleImputer

# df = some DataFrame with missing values in it
# defaults to overwriting input columns and using mean imputation
simple_imputer = SimpleImputer(input_cols=['col1'])
df_imputed_mean = simple_imputer.impute(df)

# a single wrapper class allows quick changing of parameters to enable quick EDA
simple_imputer.output_cols = ['col1_out']
simple_imputer.strategy = "median"

df_imputed_median = simple_imputer.impute(df)

# switch to imputing by forward fill
simple_imputer.input_cols = ['col2']
simple_imputer.output_cols = ['col2_out']
# must specify ordering since spark DataFrames do not have a meaningful ordering
simple_imputer.order_by_cols = ['date']
simple_imputer.strategy = "ffill"

df_imputed_ffill = simple_imputer.impute(df)

# impute multiple columns
simple_imputer.input_cols = ['col2', 'col3']
simple_imputer.output_cols = ['col2_out', 'col3_out']
simple_imputer.order_by_cols = ['date', 'date']

df_imputed_ffill_mult = simple_imputer.impute(df)
```

#### ModeImputer
All the imputation methods (besides linear regression imputation) operate fundamentally the
same with their optional arguments.

```python
from src.main.imputation.mode_imputer import ModeImputer
=======
```

#### ModeImputer
All the imputation methods (besides linear regression imputation) operate fundamentally the
same with their optional arguments.

```python
from src.main.imputation.mode_imputer import ModeImputer

# df = some DataFrame with missing values in it
mode_imputer = ModeImputer(input_cols=['col1', 'col2'], output_cols=['col1_out', 'col2_out'],
                           list_group_by_cols=[['category1', 'category2'],
                                               ['category2', 'category3']])

df_imputed_mode = mode_imputer.impute(df)
```

#### LinearRegressionImputer
```python
from src.main.imputation.linear_regression_imputer import LinearRegressionImputer
from pyspark.ml.feature import VectorAssembler

# df = some DataFrame with missing values in it
# create features column for linear regression to use
assembler = VectorAssembler(inputCols=['feature1', 'feature2', 'feature3'], outputCol='my_features')
df = assembler.transform(df)

# can choose to store summary statistics after imputation is performed
# as LinearRegressionSummary object from Spark ML.
lr_imputer = LinearRegressionImputer(features_col='my_features', col_to_impute='target_col',   
                                     store_summary_statistics=True)              

lr_imputer.impute(df)
```

### Suggested Further Work & Optimizations
1. Mean, median, mode, forward fill, backward fill
    1. Rework the code a bit to allow the usage of `HasMissingValue` from `imputer_params.py` 
    in all methods and circumstances.
    2. Minimize the number of `GROUP BY`, `ORDER BY` and `PARTITION BY` operations by 
    grouping columns that are being imputed that share identical group/order/paritition by
    parameters. 
    3. Create optional summary statistics for before-and-after imputations are created.
    4. I'm not sure if this one will work, but there may be a way to to remove the `JOIN` 
    used in mean/median/mode when group by's are used with some nested dictionary approach.
    This would be *very* beneficial as `JOIN` is an expensive operation. 
    
2. Linear regression imputer
    1. Allow more flexibility in how parameters are specified for the linear regression.
    Copying the exact parameters from LinearRegression in Spark ML may be a good idea.
    2. Allow specifying the output column for the imputation as opposed to always overwriting.
    
3. Implement more imputation methods (bolded are recent research papers)
    1. General Linear Models
    2. Stochastic regression
    3. Hot/Cold Deck
    4. K-Nearest Neighbours
    5. **Distributed Neural Networks**
    6. **Iterative Fuzzy Clustering**
    7. **Fuzzy C-means**
    8. **Association-Rules-Based Imputation**

## Feature Encoding
A package for encoding a high-dimensional categorical feature using Supervised Ratio and Weight of Evidence Encoding techniques.


There are 2 encoding methods implemented so far: `SR`, `WOE`.

In addition to those, 
there is `FeatureEncoder` which is a flexible class that acts as a wrapper around 
these 2  methods in their most simple forms. 

FeatureEncoder should be used primarily in the Data Science pipeline for Data Pre-processing such as normalizing the categorical features, encoding features with high dimension and feature engineering.

### Usage
#### FeatureEncoder
```python
# Import the class FeatureEncoder from FeatureEncoder package
from FeatureEncoder import FeatureEncoder
# Use help() on the class to understand its usage, parameters and options
help(FeatureEncoder)
```

#### Supervised Ratio Encoding
-In the supervised Machine Learning context, where class or target variables are available.
-High cardinality categorical attribute values can be converted to numerical values.
-The numerical value is a function of number of records with the categorical value for the feature and
how they break down between positive and negative class attribute values.

```python
#Create dataset
schema = StructType([StructField("AccountType", StringType(),True), 
                     StructField("EliteMember", StringType(),True), ])
data = [("Savings","Yes"),("Savings","No"),("Current","Yes"),("Current","No"),("Current","Yes")]
df = spark.createDataFrame(data, schema=schema)
#Set the Feature Encoder Class 
encoder = FeatureEncoder.FeatureEncoder(strategy = "SR" ,cat_col= "AccountType" ,label_col="EliteMember",positive_class="Yes" ,negative_class="No", bias=True)
df = encoder.encode(spark_dat)
df.show()

#Input df
+-----------+-----------+
|AccountType|EliteMember|
+-----------+-----------+
|    Savings|        Yes|
|    Savings|         No|
|    Current|        Yes|
|    Current|         No|
|    Current|        Yes|
+-----------+-----------+

#Output df
+-----------+-----------+-------------------+
|AccountType|EliteMember|AccountType_encoded|
+-----------+-----------+-------------------+
|    Savings|        Yes|0.46912058721659894|
|    Savings|         No|0.46912058721659894|
|    Current|        Yes| 0.6280003470746804|
|    Current|         No| 0.6280003470746804|
|    Current|        Yes| 0.6280003470746804|
+-----------+-----------+-------------------+

=======
#### Weight of Evidence Encoding
-Class or target variables are available.
-High cardinality categorical attribute values can be converted to numerical values.
-The numerical value is a function of number of records with the categorical value for the feature and how they break down between positive and negative class attribute values.
-Total number of records with the positive and negative class labels are also taken into account.


```python
#Create dataset
schema = StructType([StructField("AccountType", StringType(),True), 
                     StructField("EliteMember", StringType(),True), ])
data = [("Savings","Yes"),("Savings","No"),("Current","Yes"),("Current","No"),("Current","Yes")]
df = spark.createDataFrame(data, schema=schema)
#Set the Feature Encoder Class 
encoder = FeatureEncoder.FeatureEncoder(strategy = "WOE" ,cat_col= "AccountType" ,label_col="EliteMember",positive_class="Yes" ,negative_class="No", bias=True)
df = encoder.encode(spark_dat)
df.show()

#Input df
+-----------+-----------+
|AccountType|EliteMember|
+-----------+-----------+
|    Savings|        Yes|
|    Savings|         No|
|    Current|        Yes|
|    Current|         No|
|    Current|        Yes|
+-----------+-----------+

#Output df
+-----------+-----------+--------------------+
|AccountType|EliteMember| AccountType_encoded|
+-----------+-----------+--------------------+
|    Savings|        Yes|-0.22592298455604895|
|    Savings|         No|-0.22592298455604895|
|    Current|        Yes| 0.08657527033082323|
|    Current|         No| 0.08657527033082323|
|    Current|        Yes| 0.08657527033082323|
+-----------+-----------+--------------------+

```
### Special Properties
1. Can handle string as well as integer labels.
2. Can prevent model assumption by providing an option to introduce noise/penalty/bias.
3. Can handle missing target Class label

### Future Work
1. Weight of Evidence can be used for Feature Engineering
2. Current strategy only handles 2-Class Problem, can be extended to handle Multi-class problem

### Suggested Further Work & Optimizations
1. Mean, median, mode, forward fill, backward fill
    1. Rework the code a bit to allow the usage of `HasMissingValue` from `imputer_params.py` 
    in all methods and circumstances.
    2. Minimize the number of `GROUP BY`, `ORDER BY` and `PARTITION BY` operations by 
    grouping columns that are being imputed that share identical group/order/paritition by
    parameters. 
    3. Create optional summary statistics for before-and-after imputations are created.
    4. I'm not sure if this one will work, but there may be a way to to remove the `JOIN` 
    used in mean/median/mode when group by's are used with some nested dictionary approach.
    This would be *very* beneficial as `JOIN` is an expensive operation. 
    
2. Linear regression imputer
    1. Allow more flexibility in how parameters are specified for the linear regression.
    Copying the exact parameters from LinearRegression in Spark ML may be a good idea.
    2. Allow specifying the output column for the imputation as opposed to always overwriting.
    
3. Implement more imputation methods (bolded are recent research papers)
    1. General Linear Models
    2. Stochastic regression
    3. Hot/Cold Deck
    4. K-Nearest Neighbours
    5. **Distributed Neural Networks**
    6. **Iterative Fuzzy Clustering**
    7. **Fuzzy C-means**
    8. **Association-Rules-Based Imputation**

### References
https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html
