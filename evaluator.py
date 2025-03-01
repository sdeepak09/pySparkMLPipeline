from pyspark.ml.evaluation import (
    RegressionEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame
import mlflow
from utils import logger


class ModelEvaluator:
    """A class for evaluating PySpark ML model predictions across regression, binary, and multiclass classification.

    Args:
        eval_metric (str): Primary evaluation metric (e.g., "rmse", "areaUnderROC", "f1").
        label_col (str, optional): Label column name. Defaults to "label".
        prediction_col (str, optional): Prediction column name. Defaults to "prediction".
        raw_prediction_col (str, optional): Raw prediction column name for binary classification. Defaults to "rawPrediction".
        metric_definitions (dict, optional): Custom metric definitions.

    Attributes:
        metric_definitions (dict): Holds default or custom metric definitions.
    """

    DEFAULT_METRICS = {
        "regression": ["rmse", "mse", "r2", "mae"],
        "binary_classification": ["areaUnderROC", "areaUnderPR"],
        "multiclass_classification": [
            "f1",
            "accuracy",
            "weightedPrecision",
            "weightedRecall",
        ],
    }

    def __init__(
        self,
        eval_metric: str,
        label_col: str = "label",
        prediction_col: str = "prediction",
        raw_prediction_col: str = "rawPrediction",
        metric_definitions: dict = None,
    ):
        self.eval_metric = eval_metric
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.raw_prediction_col = raw_prediction_col

        # Determine problem type based on eval_metric
        if eval_metric in self.DEFAULT_METRICS["regression"]:
            self.problem_type = "regression"
        elif eval_metric in self.DEFAULT_METRICS["binary_classification"]:
            self.problem_type = "binary_classification"
        elif eval_metric in self.DEFAULT_METRICS["multiclass_classification"]:
            self.problem_type = "multiclass_classification"
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")

        # Use provided metric definitions or default ones
        self.metric_definitions = metric_definitions or self._get_default_metrics()

    def _get_default_metrics(self):
        """Returns the default metric definitions based on the problem type."""
        if self.problem_type == "regression":
            return {
                metric: {"evaluator": "RegressionEvaluator", "params": {}}
                for metric in self.DEFAULT_METRICS["regression"]
            }
        elif self.problem_type == "binary_classification":
            return {
                metric: {"evaluator": "BinaryClassificationEvaluator", "params": {}}
                for metric in self.DEFAULT_METRICS["binary_classification"]
            }
        elif self.problem_type == "multiclass_classification":
            return {
                metric: {"evaluator": "MulticlassClassificationEvaluator", "params": {}}
                for metric in self.DEFAULT_METRICS["multiclass_classification"]
            }
        return {}

    def get_evaluator(self, metric_name: str):
        """Returns the appropriate PySpark Evaluator for the given metric."""
        metric_info = self.metric_definitions.get(metric_name)
        if not metric_info:
            raise ValueError(f"Unsupported metric: {metric_name}")

        evaluator_type = metric_info["evaluator"]
        params = metric_info["params"]

        if evaluator_type == "RegressionEvaluator":
            return RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName=metric_name,
                **params,
            )
        elif evaluator_type == "BinaryClassificationEvaluator":
            return BinaryClassificationEvaluator(
                labelCol=self.label_col,
                rawPredictionCol=self.raw_prediction_col,
                metricName=metric_name,
                **params,
            )
        elif evaluator_type == "MulticlassClassificationEvaluator":
            return MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName=metric_name,
                **params,
            )
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

    def evaluate_predictions(self, predictions: DataFrame) -> dict:
        """Evaluates the predictions and calculates the relevant metrics.

        Instead of failing on the first error, this version logs errors and continues evaluation.
        """
        results = {}
        for metric_name in self.metric_definitions.keys():
            try:
                evaluator = self.get_evaluator(metric_name)
                score = evaluator.evaluate(predictions)
                results[metric_name] = score
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
                results[metric_name] = (
                    None  # Record failure without aborting all evaluations
                )
        return results

    def log_metrics(self, metrics: dict, use_mlflow: bool = False):
        """Logs the calculated metrics using the logger and optionally logs them to MLflow."""
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value}")
            if use_mlflow:
                mlflow.log_metric(metric_name, value)
