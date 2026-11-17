
from pathlib import Path
import json

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "cleaned_claims"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression"
RESULT_PATH = BASE_DIR / "outputs" / "evaluation_results.json"

spark = (
    SparkSession.builder
    .appName("InsuranceClaimModel")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nINSURANCE CLAIM MACHINE LEARNING")
print("=" * 45)

# Load cleaned data
data = spark.read.csv(
    str(DATA_PATH),
    header=True,
    inferSchema=True
)

print("Total records:", data.count())

# Encode insurance provider
provider_indexer = StringIndexer(
    inputCol="insurance_provider",
    outputCol="provider_index",
    handleInvalid="keep"
)

provider_encoder = OneHotEncoder(
    inputCols=["provider_index"],
    outputCols=["provider_encoded"]
)

# Encode target variable
label_indexer = StringIndexer(
    inputCol="claim_status",
    outputCol="label"
)

# Combine features
assembler = VectorAssembler(
    inputCols=[
        "provider_encoded",
        "billed_amount",
        "claim_month",
        "claim_year"
    ],
    outputCol="features"
)

# Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50
)

# Build pipeline
pipeline = Pipeline(
    stages=[
        provider_indexer,
        provider_encoder,
        label_indexer,
        assembler,
        lr
    ]
)

# Train-test split
train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records:", train_data.count())
print("Testing records:", test_data.count())

# Train
print("\nTraining Logistic Regression...")
model = pipeline.fit(train_data)

# Predictions
predictions = model.transform(test_data)

print("\nSample predictions:")
predictions.select(
    "claim_status",
    "prediction",
    "probability"
).show(10, truncate=False)

# Evaluate
metrics = {}

for metric in ["accuracy", "weightedPrecision",
               "weightedRecall", "f1"]:

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric
    )

    score = evaluator.evaluate(predictions)

    metrics[metric] = round(score, 6)

    print(f"{metric}: {score:.4f}")

# Save model
model.write().overwrite().save(str(MODEL_PATH))

# Save evaluation results
with open(RESULT_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Model path:", MODEL_PATH)
print("Metrics saved:", RESULT_PATH)

spark.stop()
