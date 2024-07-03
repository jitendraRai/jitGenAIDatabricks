# Databricks notebook source
# MAGIC %md
# MAGIC ### Data Preprocessing steps
# MAGIC ** **
# MAGIC In this notebook we execute some preprocessing steps to get the wine dataset ready for analysis and training. After the preprocessing steps, we update the created train and test tables identified by your name. Still there are some preprocessing steps left to be done, so inspect the dataframe and its columns to see if you can improve the resulting train and test data. Improving the data will result in better model performances!

# COMMAND ----------

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

data_path = 'data/wine_first_batch.csv'
my_name = 'FILL_IN'

# COMMAND ----------

zip_file_path = 'data/wine_data.zip'
# Extract the zip file
if not os.path.exists(data_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('data')

# COMMAND ----------

# MAGIC %md
# MAGIC First we read the first batch data into a Spark dataframe and view the amount of rows and columns.

# COMMAND ----------

df =  spark.read.option("delimiter", ",").option("header", True).csv(f"file:{os.getcwd()}/{data_path}")
df.display()

# COMMAND ----------

print((df.count(), len(df.columns)))
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC As a first preprocessing step, we create two separate columns out of the appellation column: country and region

# COMMAND ----------

df = (df.withColumn("country", F.trim(F.element_at(F.split(F.col("appellation"), ","), -1)))
                    .withColumn("region", F.trim(F.element_at(F.split(F.col("appellation"), ","), -2)))
                  )
df = df.filter(df.region.contains("$") == False)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC For the train and test split, we need to add an id column. First, we drop the null values in the alcohol column. We also remove the % sign from the alcohol and the $ sign from the price and convert them to decimals.

# COMMAND ----------

df = df.na.drop(subset = ['alcohol'])
df = (df.withColumn("id", F.monotonically_increasing_id().cast(LongType()))
        .withColumn('alcohol', F.regexp_replace('alcohol', '%', '').cast(DecimalType(20,1)))
        .withColumn('price', F.regexp_replace('price', '\$', '').cast(DecimalType(20,0)))
        .withColumnRenamed('varietal', 'grape_variety')
        .withColumnRenamed('designation', 'wine_name')
        .withColumn('rating', df.rating.cast(IntegerType()))
        .withColumn('year', df.year.cast(IntegerType()))
)

# Get the list of columns
columns = df.columns

# Reorder columns: move 'ID' to the front
reordered_columns = ['id'] + [col for col in columns if col != 'id']

# Select reordered columns
df = df.select(reordered_columns)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we split the preprocessed wine data into train and test dataframes. We use the same seed of 42 to ensure similar experiments between groups in the workshop.

# COMMAND ----------

train, test = df.randomSplit([0.8,0.2], seed=42)
test.display()

# COMMAND ----------

train.createOrReplaceTempView('train_data')
test.createOrReplaceTempView('test_data')

# COMMAND ----------

# MAGIC %md
# MAGIC The final step is to merge preprocessed dataframe into a table uniquely identified by your name.

# COMMAND ----------

spark.sql(f"use db_{my_name}")

spark.sql(f"""
MERGE INTO db_{my_name}.wine_train_data_{my_name}
USING train_data
ON wine_train_data_{my_name}.id = train_data.id
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED THEN
  INSERT *
""")

spark.sql(f"""
MERGE INTO db_{my_name}.wine_test_data_{my_name}
USING test_data
ON wine_test_data_{my_name}.id = test_data.id
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED THEN
  INSERT *
""")

# COMMAND ----------

print((train.count(), len(train.columns)))
print((test.count(), len(test.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC We have usuable train and test tables to use for experiments! Now create an experiment with the train data using AutoML and find the best performing (least mean-absolute error) model!
# MAGIC
