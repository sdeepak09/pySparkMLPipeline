from abc import ABC, abstractmethod
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.regression import GBTRegressor
import os
from utils import logger

try:
    from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning(
        "XGBoost is not installed. Please install xgboost for Spark support."
    )
    XGBOOST_AVAILABLE = False


class BaseModel(ABC):
    """Abstract base class for PySpark ML models.

    Args:
        feature_cols (list[str]): List of feature column names.
        label_col (str, optional): Name of the label column. Defaults to "label".
        prediction_col (str, optional): Name of the prediction column. Defaults to "prediction".
        **kwargs: Additional parameters passed to the specific ML model.
    """

    def __init__(
        self,
        feature_cols: list[str],
        label_col: str = "label",
        prediction_col: str = "prediction",
        **kwargs,
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.prediction_col = prediction_col
        self._is_fitted = False
        self.model = self.get_model(**kwargs)

    @abstractmethod
    def get_model(self, **kwargs):
        """Returns an instance of a PySpark ML model."""
        pass

    def fit(self, train_df: DataFrame):
        """Fits the model to the training data."""
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")
        pipeline = Pipeline(stages=[assembler, self.model])
        self.pipeline_model = pipeline.fit(train_df)
        self._is_fitted = True
        logger.info("Model has been successfully fitted.")
        return self

    def predict(self, test_df: DataFrame) -> DataFrame:
        """Generates predictions using the trained model."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call `fit` before predicting.")
        predictions = self.pipeline_model.transform(test_df)
        return predictions

    def get_params(self) -> dict:
        """Returns the hyperparameters of the trained model.

        Note: This method assumes that the model is the last stage in the pipeline.
        """
        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Call `fit` before retrieving parameters."
            )
        return {
            param.name: self.pipeline_model.stages[-1].getOrDefault(param)
            for param in self.pipeline_model.stages[-1].extractParamMap()
        }

    def save(self, path: str):
        """Saves the trained model to the specified path."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Cannot save an unfitted model.")
        self.pipeline_model.write().overwrite().save(path)
        logger.info(f"Model saved successfully at {path}")

    @classmethod
    def load(
        cls,
        path: str,
        feature_cols: list[str],
        label_col: str = "label",
        prediction_col: str = "prediction",
    ):
        """Loads a trained model from the specified path."""
        try:
            pipeline_model = Pipeline.load(path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model from {path}: {str(e)}")

        model_instance = pipeline_model.stages[-1]

        model_class = None
        for subclass in cls.__subclasses__():
            if hasattr(subclass, "_model_cls") and isinstance(
                model_instance, subclass._model_cls
            ):
                model_class = subclass
                break

        if model_class is None:
            raise ValueError(
                "Cannot determine the appropriate model class for loading. Ensure that the subclass defines _model_cls correctly."
            )

        instance = model_class(feature_cols, label_col, prediction_col)
        instance.pipeline_model = pipeline_model
        instance._is_fitted = True
        logger.info(f"Model loaded successfully from {path}")
        return instance


class LogisticRegressionModel(BaseModel):
    """Logistic Regression Model."""

    _model_cls = LogisticRegression

    def get_model(self, **kwargs):
        return LogisticRegression(
            labelCol=self.label_col,
            featuresCol="features",
            predictionCol=self.prediction_col,
            **kwargs,
        )


class RandomForestModel(BaseModel):
    """Random Forest Model."""

    _model_cls = RandomForestClassifier

    def get_model(self, **kwargs):
        return RandomForestClassifier(
            labelCol=self.label_col,
            featuresCol="features",
            predictionCol=self.prediction_col,
            **kwargs,
        )


class GradientBoostedTreeModel(BaseModel):
    """Gradient Boosted Trees Model."""

    _model_cls = GBTClassifier

    def get_model(self, **kwargs):
        return GBTClassifier(
            labelCol=self.label_col,
            featuresCol="features",
            predictionCol=self.prediction_col,
            **kwargs,
        )


class GradientBoostedTreeRegressorModel(BaseModel):
    """Gradient Boosted Trees for Regression."""

    _model_cls = GBTRegressor

    def get_model(self, **kwargs):
        return GBTRegressor(
            labelCol=self.label_col,
            featuresCol="features",
            predictionCol=self.prediction_col,
            **kwargs,
        )


class XGBoostModel(BaseModel):
    """XGBoost Model (if XGBoost is installed)."""

    if XGBOOST_AVAILABLE:
        _model_cls = SparkXGBClassifier
    else:
        _model_cls = None

    def get_model(self, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install the 'xgboost' package."
            )
        return SparkXGBClassifier(
            labelCol=self.label_col,
            featuresCol="features",
            predictionCol=self.prediction_col,
            **kwargs,
        )
