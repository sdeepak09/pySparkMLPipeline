import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import (
    RegressionEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import optuna
import mlflow
import mlflow.spark
from pyspark.sql import DataFrame
from utils import logger


class HyperparameterTuner:
    """Hyperparameter tuner using either Hyperopt or Optuna for PySpark ML models.

    Args:
        optimizer_backend (str): The optimization backend to use ("hyperopt" or "optuna").
    """

    def __init__(self, optimizer_backend: str = "hyperopt"):
        if optimizer_backend not in ["hyperopt", "optuna"]:
            raise ValueError(
                "Unsupported optimizer backend. Choose 'hyperopt' or 'optuna'."
            )
        self.optimizer_backend = optimizer_backend

    def tune_model(
        self,
        train_df: DataFrame,
        model_class,
        search_space: dict,
        eval_metric: str,
        feature_cols: list[str],
        label_col: str = "label",
        prediction_col: str = "prediction",
        num_evals: int = 100,
        use_cv: bool = False,
        cv_folds: int = 3,
        use_mlflow: bool = False,
        mlflow_experiment_name: str = None,
    ) -> dict:
        """Tunes the model using the specified optimizer backend."""
        if self.optimizer_backend == "hyperopt":
            return self._tune_with_hyperopt(
                train_df,
                model_class,
                search_space,
                eval_metric,
                feature_cols,
                label_col,
                prediction_col,
                num_evals,
                use_cv,
                cv_folds,
                use_mlflow,
                mlflow_experiment_name,
            )
        else:
            return self._tune_with_optuna(
                train_df,
                model_class,
                search_space,
                eval_metric,
                feature_cols,
                label_col,
                prediction_col,
                num_evals,
                use_cv,
                cv_folds,
                use_mlflow,
                mlflow_experiment_name,
            )

    def _tune_with_hyperopt(
        self,
        train_df,
        model_class,
        search_space,
        eval_metric,
        feature_cols,
        label_col,
        prediction_col,
        num_evals,
        use_cv,
        cv_folds,
        use_mlflow,
        mlflow_experiment_name,
    ):
        """Hyperopt-based tuning implementation."""
        translated_space = self._translate_search_space(search_space)

        def objective(params):
            # Split the provided data into training and validation subsets
            train_split, valid_split = train_df.randomSplit([0.8, 0.2], seed=123)

            model = model_class(
                feature_cols=feature_cols,
                label_col=label_col,
                prediction_col=prediction_col,
                **params,
            )
            evaluator = self.get_evaluator(eval_metric, label_col, prediction_col)
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            pipeline = Pipeline(stages=[assembler, model])
            pipeline_model = pipeline.fit(train_split)
            predictions = pipeline_model.transform(valid_split)
            score = evaluator.evaluate(predictions)
            if use_mlflow:
                mlflow.log_params(params)
                mlflow.log_metric(eval_metric, score)
            return {"loss": -score, "status": STATUS_OK}

        best_params = fmin(
            fn=objective,
            space=translated_space,
            algo=tpe.suggest,
            max_evals=num_evals,
            trials=Trials(),
        )
        logger.info(f"Best Hyperopt Parameters: {best_params}")
        return best_params

    def _tune_with_optuna(
        self,
        train_df,
        model_class,
        search_space,
        eval_metric,
        feature_cols,
        label_col,
        prediction_col,
        num_evals,
        use_cv,
        cv_folds,
        use_mlflow,
        mlflow_experiment_name,
    ):
        """Optuna-based tuning implementation."""

        def objective(trial):
            params = {}
            for param_name, param_def in search_space.items():
                if param_def["type"] == "uniform":
                    params[param_name] = trial.suggest_uniform(
                        param_name, param_def["low"], param_def["high"]
                    )
                elif param_def["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_def["low"],
                        param_def["high"],
                        step=param_def.get("step", 1),
                    )
                elif param_def["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_def["values"]
                    )
                elif param_def["type"] == "loguniform":
                    params[param_name] = trial.suggest_loguniform(
                        param_name, param_def["low"], param_def["high"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {param_def['type']}"
                    )
            # Split into training and validation sets
            train_split, valid_split = train_df.randomSplit([0.8, 0.2], seed=123)

            model = model_class(
                feature_cols=feature_cols,
                label_col=label_col,
                prediction_col=prediction_col,
                **params,
            )
            evaluator = self.get_evaluator(eval_metric, label_col, prediction_col)
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            pipeline = Pipeline(stages=[assembler, model])
            pipeline_model = pipeline.fit(train_split)
            predictions = pipeline_model.transform(valid_split)
            score = evaluator.evaluate(predictions)
            if use_mlflow:
                mlflow.log_params(params)
                mlflow.log_metric(eval_metric, score)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_evals)
        logger.info(f"Best Optuna Parameters: {study.best_params}")
        return study.best_params

    def get_evaluator(
        self,
        eval_metric: str,
        label_col: str = "label",
        prediction_col: str = "prediction",
    ):
        """Returns the appropriate PySpark evaluator based on the metric."""
        if eval_metric in ["rmse", "mse", "r2", "mae"]:
            return RegressionEvaluator(
                labelCol=label_col, predictionCol=prediction_col, metricName=eval_metric
            )
        elif eval_metric in ["areaUnderROC", "areaUnderPR"]:
            # For binary classification, use the standard rawPredictionCol
            return BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
                metricName=eval_metric,
            )
        elif eval_metric in ["f1", "accuracy", "weightedPrecision", "weightedRecall"]:
            return MulticlassClassificationEvaluator(
                labelCol=label_col, predictionCol=prediction_col, metricName=eval_metric
            )
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")

    def _translate_search_space(self, search_space: dict) -> dict:
        """Converts custom search space into Hyperopt-compatible format."""
        translated_space = {}
        for param_name, param_def in search_space.items():
            if param_def["type"] == "uniform":
                translated_space[param_name] = hp.uniform(
                    param_name, param_def["low"], param_def["high"]
                )
            elif param_def["type"] == "int":
                translated_space[param_name] = hp.quniform(
                    param_name,
                    param_def["low"],
                    param_def["high"],
                    param_def.get("step", 1),
                )
            elif param_def["type"] == "categorical":
                translated_space[param_name] = hp.choice(
                    param_name, param_def["values"]
                )
            elif param_def["type"] == "loguniform":
                translated_space[param_name] = hp.loguniform(
                    param_name, param_def["low"], param_def["high"]
                )
            else:
                raise ValueError(f"Unsupported search space type: {param_def['type']}")
        return translated_space
