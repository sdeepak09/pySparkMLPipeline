from enum import Enum
from pydantic import BaseModel, validator


class FileFormat(str, Enum):
    PARQUET = "parquet"
    CSV = "csv"


class DataConfig(BaseModel):
    """
    Configuration model for specifying S3 data reading parameters.

    Attributes:
        s3_bucket (str): The name of the S3 bucket.
        data_path (str): The relative path to the data file or directory in the S3 bucket.
        file_format (FileFormat): The format of the data file (default: "parquet").
    Note:
        Additional validations (e.g., bucket naming conventions) can be added as needed.
    """

    s3_bucket: str
    data_path: str
    file_format: FileFormat = FileFormat.PARQUET

    @validator("s3_bucket", "data_path")
    def not_empty(cls, v, field):
        if not v:
            raise ValueError(f"{field.name} cannot be empty")
        return v
