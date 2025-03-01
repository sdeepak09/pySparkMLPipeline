import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, rand
from config import DataConfig
from utils import logger, DataReader
from tuner import HyperparameterTuner
from models import LogisticRegressionModel, RandomForestModel
from evaluator import ModelEvaluator
from selector import ModelSelector


def stratified_split(df, label_col, train_fraction=0.8, seed=123):
    """Performs stratified splitting of a PySpark DataFrame."""
    # Add a temporary column with random values for proper stratification
    stratified_df = df.withColumn("rand", rand(seed))

    # Compute sampling fractions for each label
    fractions = {
        row[label_col]: train_fraction
        for row in stratified_df.select(label_col).distinct().collect()
    }
    train_df = stratified_df.sampleBy(label_col, fractions=fractions, seed=seed)
    test_df = stratified_df.subtract(train_df)

    # Drop the temporary column if it exists
    if "rand" in train_df.columns:
        train_df = train_df.drop("rand")
    if "rand" in test_df.columns:
        test_df = test_df.drop("rand")
    return train_df, test_df


def main():
    """Main function to orchestrate the end-to-end PySpark ML pipeline."""
    # -----------------------
    # Step 1: Configuration
    # -----------------------
    s3_bucket = "your_bucket"  # Change this to your actual S3 bucket
    data_path = "your_data.parquet"
    file_format = "parquet"  # For production, consider externalizing configuration

    tune_hyperparams = True  # Set False to use predefined hyperparameters
    optimizer_backend = "optuna"  # Choose between "hyperopt" or "optuna"
    num_evals = 20  # Number of tuning trials

    eval_metric = "areaUnderROC"  # Metric for selecting best model
    higher_is_better = True  # True if higher metric values indicate better performance

    feature_cols = ["feature1", "feature2", "feature3"]
    label_col = "label"

    model_save_path = "/tmp/best_model"  # Path to save the best model

    models_to_train = [
        {
            "model_class": LogisticRegressionModel,
            "model_name": "lr_tuned",
            "hyperparams": None,
        },
        {
            "model_class": RandomForestModel,
            "model_name": "rf_default",
            "hyperparams": {"numTrees": 50, "maxDepth": 10},
        },
    ]

    search_space_hyperopt = {
        "regParam": {"type": "uniform", "low": 0.01, "high": 0.5},
        "elasticNetParam": {"type": "loguniform", "low": 1e-5, "high": 1.0},
        "maxIter": {"type": "int", "low": 10, "high": 50, "step": 10},
        "fitIntercept": {"type": "categorical", "values": [True, False]},
    }

    # -----------------------
    # Step 2: Initialize Spark & Read Data
    # -----------------------
    spark = SparkSession.builder.appName("MLPipeline").getOrCreate()
    data_config = DataConfig(
        s3_bucket=s3_bucket, data_path=data_path, file_format=file_format
    )
    reader = DataReader(spark)
    df = reader.read_data(data_config)
    if df is None:
        logger.error(
            f"Data loading failed for S3 path: s3://{data_config.s3_bucket}/{data_config.data_path}. Exiting pipeline."
        )
        spark.stop()
        return

    train_df, test_df = stratified_split(df, label_col, train_fraction=0.8, seed=123)

    # -----------------------
    # Step 3: Train Models
    # -----------------------
    trained_models = {}
    model_metrics = {}

    for model_cfg in models_to_train:
        model_class = model_cfg["model_class"]
        model_name = model_cfg["model_name"]
        hyperparams = model_cfg["hyperparams"]

        logger.info(f"Training model: {model_name}")

        if tune_hyperparams and hyperparams is None:
            tuner = HyperparameterTuner(optimizer_backend=optimizer_backend)
            best_params = tuner.tune_model(
                train_df=train_df,
                model_class=model_class,
                search_space=search_space_hyperopt,
                eval_metric=eval_metric,
                feature_cols=feature_cols,
                num_evals=num_evals,
            )
            logger.info(f"Best params for {model_name}: {best_params}")
        else:
            best_params = hyperparams

        # Instantiate and train the model with required parameters and hyperparameters
        model_instance = model_class(
            feature_cols=feature_cols,
            label_col=label_col,
            prediction_col="prediction",
            **(best_params or {}),
        )
        model_instance.fit(train_df)
        trained_models[model_name] = model_instance

    # -----------------------
    # Step 4: Evaluate Models
    # -----------------------
    evaluator = ModelEvaluator(eval_metric=eval_metric)
    for model_name, model_instance in trained_models.items():
        logger.info(f"Evaluating model: {model_name}")
        predictions = model_instance.predict(test_df)
        metrics = evaluator.evaluate_predictions(predictions)
        model_metrics[model_name] = metrics
        logger.info(f"Metrics for {model_name}: {metrics}")

    # -----------------------
    # Step 5: Select the Best Model
    # -----------------------
    selector = ModelSelector()
    best_model_id = selector.select_best_model(
        model_metrics, eval_metric, higher_is_better
    )
    logger.info(f"Best model selected: {best_model_id}")

    # -----------------------
    # Step 6: Save the Best Model
    # -----------------------
    artifacts = {"feature_cols": feature_cols}
    selector.save_best_model(
        best_model_identifier=best_model_id,
        models_dict=trained_models,
        save_path=model_save_path,
        artifacts=artifacts,
        eval_metric=eval_metric,
        model_metrics=model_metrics,
    )
    logger.info(f"Best model saved successfully at: {model_save_path}")
    spark.stop()


if __name__ == "__main__":
    main()
