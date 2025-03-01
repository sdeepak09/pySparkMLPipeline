import logging
from pyspark.sql import SparkSession
from pyspark.errors import PySparkException
from botocore.exceptions import NoCredentialsError
from config import DataConfig, FileFormat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataReader:
    """
    Utility class for reading data from an S3 bucket into a PySpark DataFrame.
    """

    def __init__(self, spark=None):
        """
        Initializes the DataReader with an existing Spark session or creates a new one.
        """
        self.spark = (
            spark
            if spark is not None
            else SparkSession.builder.appName("S3DataReader").getOrCreate()
        )

    def read_data(self, config: DataConfig):
        """
        Reads data from an S3 bucket into a PySpark DataFrame.
        """
        s3_path = f"s3://{config.s3_bucket}/{config.data_path}"
        logger.info(f"Attempting to read data from {s3_path} as {config.file_format}")

        try:
            if config.file_format == FileFormat.PARQUET:
                df = self.spark.read.parquet(s3_path)
            elif config.file_format == FileFormat.CSV:
                # Added inferSchema and default delimiter for CSV reading
                df = (
                    self.spark.read.option("header", "true")
                    .option("inferSchema", "true")
                    .csv(s3_path)
                )
            else:
                raise ValueError(f"Unsupported file format: {config.file_format}")

            logger.info(f"Successfully read data from {s3_path}")
            return df

        except FileNotFoundError:
            logger.error(f"FileNotFoundError: Could not find file at {s3_path}")
        except NoCredentialsError:
            logger.error(
                "No AWS credentials found. Ensure you have set up credentials."
            )
        except PySparkException as e:
            logger.error(f"PySparkException: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while reading {s3_path}: {str(e)}")
        return None
