import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    month,
    year,
    count
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "dataset" / "claims_and_billing.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "cleaned_claims"

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("InsuranceClaimPrediction")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nINSURANCE CLAIM DATA PROCESSING")
print("=" * 45)

# 1. Load dataset
data = spark.read.csv(
    str(INPUT_PATH),
    header=True,
    inferSchema=True
)

print("\nOriginal record count:", data.count())

print("\nDataset Schema:")
data.printSchema()

# 2. Missing value analysis
print("\nMissing Values:")

data.select([
    count(
        col(c).isNull().cast("int")
    ).alias(c)
    for c in data.columns
]).show(truncate=False)

# 3. Filter insurance payments
data = data.filter(
    col("payment_method") == "Insurance"
)

# 4. Remove invalid records
data = data.filter(
    col("claim_id").isNotNull()
    & col("claim_billing_date").isNotNull()
    & col("insurance_provider").isNotNull()
    & col("claim_status").isin("Paid", "Denied")
    & (col("billed_amount") > 0)
)

# 5. Convert billing date
data = data.withColumn(
    "claim_billing_date",
    to_timestamp(
        col("claim_billing_date"),
        "dd-MM-yyyy HH:mm"
    )
)

data = data.filter(
    col("claim_billing_date").isNotNull()
)

# 6. Feature engineering
data = data.withColumn(
    "claim_month",
    month(col("claim_billing_date"))
)

data = data.withColumn(
    "claim_year",
    year(col("claim_billing_date"))
)

# 7. Display cleaned results
print("\nCleaned record count:", data.count())

print("\nClaim Status Distribution:")

data.groupBy("claim_status").count().show()

print("\nInsurance Provider Summary:")

data.groupBy(
    "insurance_provider",
    "claim_status"
).count().orderBy(
    "insurance_provider"
).show()

# 8. Save cleaned dataset
data.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(str(OUTPUT_PATH))

print("\nCleaned dataset saved successfully!")
print("Output location:", OUTPUT_PATH)

spark.stop()
