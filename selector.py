import json
import os
from datetime import datetime
from utils import logger


class ModelSelector:
    """A utility class for selecting the best model based on evaluation metrics and saving it."""

    def select_best_model(
        self, model_metrics: dict, eval_metric: str, higher_is_better: bool = False
    ) -> str:
        """Selects the best model based on the specified evaluation metric."""
        if not model_metrics:
            raise ValueError(
                "The `model_metrics` dictionary is empty. Cannot determine the best model."
            )

        if not any(eval_metric in metrics for metrics in model_metrics.values()):
            raise ValueError(
                f"Evaluation metric '{eval_metric}' not found in any model's metrics."
            )

        best_model = None
        best_score = None
        for model_id, metrics in model_metrics.items():
            # Skip if the evaluation metric is missing or None for a model
            if eval_metric not in metrics or metrics[eval_metric] is None:
                logger.warning(
                    f"Model '{model_id}' does not have a valid '{eval_metric}' metric. Skipping."
                )
                continue
            score = metrics[eval_metric]
            if (
                best_model is None
                or (higher_is_better and score > best_score)
                or (not higher_is_better and score < best_score)
            ):
                best_model = model_id
                best_score = score

        if best_model is None:
            raise ValueError(
                "No valid model was found based on the given evaluation metric."
            )
        logger.info(
            f"Best model selected: {best_model} with {eval_metric} = {best_score}"
        )
        return best_model

    def save_best_model(
        self,
        best_model_identifier: str,
        models_dict: dict,
        save_path: str,
        artifacts: dict = None,
        eval_metric: str = None,
        model_metrics: dict = None,
    ):
        """Saves the best model along with relevant metadata and artifacts."""
        if best_model_identifier not in models_dict:
            raise ValueError(
                f"Best model '{best_model_identifier}' not found in `models_dict`."
            )

        best_model = models_dict[best_model_identifier]
        model_save_path = os.path.join(save_path, "model")
        artifacts_path = os.path.join(save_path, "artifacts")
        metadata_path = os.path.join(save_path, "metadata.json")

        try:
            os.makedirs(model_save_path, exist_ok=True)
            os.makedirs(artifacts_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise e

        best_model.save(model_save_path)

        if artifacts:
            try:
                artifacts_file = os.path.join(artifacts_path, "artifacts.json")
                with open(artifacts_file, "w") as f:
                    json.dump(artifacts, f, indent=4)
                logger.info(f"Artifacts saved at {artifacts_file}")
            except Exception as e:
                logger.error(f"Error saving artifacts: {str(e)}")

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "best_model_id": best_model_identifier,
            "eval_metric": eval_metric,
            "all_metrics": model_metrics,
        }
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved at {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
        logger.info(f"Best model saved at {model_save_path}")
