# =============================================================
# Retail Supply Chain Optimization using Apache Spark and ML
# Author: Viraj Indais
# Tech Stack: PySpark, Spark MLlib, Spark SQL, Google Colab
# =============================================================

# -- 1. INITIALIZE SPARK SESSION ------------------------------
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Retail Supply Chain Optimization") \
    .getOrCreate()

print("Spark Session Initialized Successfully")
print("Spark Version:", spark.version)

# -- 2. LOAD DATASETS -----------------------------------------
# Dataset: Walmart Retail Sales Dataset (Kaggle)
# Files: sales data-set.csv | Features data set.csv | stores data-set.csv

sales_df    = spark.read.csv("sales data-set.csv",    header=True, inferSchema=True)
features_df = spark.read.csv("Features data set.csv", header=True, inferSchema=True)
stores_df   = spark.read.csv("stores data-set.csv",   header=True, inferSchema=True)

print("\nDatasets Loaded:")
print("Sales Records    :", sales_df.count(), "rows")
print("Features Records :", features_df.count(), "rows")
print("Store Records    :", stores_df.count(), "rows")

sales_df.show(5)

# -- 3. DISTRIBUTED JOIN OPERATIONS ---------------------------
sales_features_df = sales_df.join(
    features_df,
    on=["Store", "Date"],
    how="inner"
)

final_joined_df = sales_features_df.join(
    stores_df,
    on="Store",
    how="inner"
)

print("\nJoined Dataset:", final_joined_df.count(), "rows")

# -- 4. DATA PREPROCESSING ------------------------------------
from pyspark.sql.functions import col

selected_df = final_joined_df.select(
    col("Store"),
    col("Dept"),
    col("Temperature").cast("double"),
    col("Fuel_Price").cast("double"),
    col("CPI").cast("double"),
    col("Unemployment").cast("double"),
    col("Weekly_Sales").cast("double")
).dropna()

print("\nAfter Preprocessing:", selected_df.count(), "rows")
selected_df.printSchema()

# -- 5. FEATURE ENGINEERING -----------------------------------
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["Temperature", "Fuel_Price", "CPI", "Unemployment"],
    outputCol="features"
)

model_df = assembler.transform(selected_df)
print("\nFeature Vector Created using VectorAssembler")

# -- 6. INVENTORY ANALYSIS ------------------------------------
from pyspark.sql.functions import sum as spark_sum

# Total sales per store
store_sales = final_joined_df.groupBy("Store") \
    .agg(spark_sum("Weekly_Sales").alias("Total_Sales"))

# Slow-moving inventory detection
avg_sales = store_sales.agg({"Total_Sales": "avg"}).collect()[0][0]

slow_inventory_df = store_sales.withColumn(
    "Slow_Moving",
    (col("Total_Sales") < avg_sales).cast("int")
)

print("\nAverage Weekly Sales per Store:", round(avg_sales, 2))
print("\nSlow-Moving Inventory Detection:")
slow_inventory_df.show(10)

# Stock Turnover Ratio
stock_turnover = store_sales.withColumn(
    "Stock_Turnover_Ratio",
    col("Total_Sales") / avg_sales
)

print("\nStock Turnover Ratio by Store:")
stock_turnover.show(10)

# -- 7. TRAIN-TEST SPLIT --------------------------------------
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

print("\nTrain Set:", train_df.count(), "rows")
print("Test Set :", test_df.count(), "rows")

# -- 8. LINEAR REGRESSION MODEL -------------------------------
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="Weekly_Sales"
)

lr_model    = lr.fit(train_df)
predictions = lr_model.transform(test_df)

print("\nModel Training Complete")
print("\nSample Predictions (Actual vs Predicted Weekly Sales):")
predictions.select("Weekly_Sales", "prediction").show(10)

# -- 9. RESTOCK DECISION LOGIC --------------------------------
from pyspark.sql.functions import when

# Business Rule: If predicted sales > current sales, restock is needed
result_df = predictions.withColumn(
    "Restock_Required",
    when(col("prediction") > col("Weekly_Sales"), 1).otherwise(0)
)

print("\nAutomated Restock Decision Logic:")
print("Rule: If Predicted Sales > Current Sales → Restock_Required = 1")
result_df.select("Weekly_Sales", "prediction", "Restock_Required").show(10)

restock_count = result_df.filter(col("Restock_Required") == 1).count()
print("Stores Requiring Restock:", restock_count)

# -- 10. MODEL EVALUATION -------------------------------------
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)

print("\n" + "=" * 45)
print("  Model Evaluation Results")
print("=" * 45)
print("  RMSE     :", round(rmse, 2))
print("  R2 Score :", round(r2, 4))
print("=" * 45)
print("\n  Note: Low R2 indicates that temperature, fuel price,")
print("  CPI, and unemployment alone are insufficient predictors.")
print("  Future work: Include promotions, holidays, store type.")

# -- END ------------------------------------------------------
spark.stop()
print("\nSpark Session Stopped.")
