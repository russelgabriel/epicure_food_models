from abc import ABC, abstractmethod
import polars as pl
from .schema import UNIFIED_SCHEMA

class BaseParser(ABC):
  
  @abstractmethod
  def parse(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Parses a raw dataset file and returns a polars DataFrame
    adhering to the UNIFIED_SCHEMA.
    """
    pass
  
  def validate_schema(self, df: pl.DataFrame) -> bool:
    """
    Validates that the DataFrame adheres to the UNIFIED_SCHEMA.
    """
    if list(df.schema.keys()) != list(UNIFIED_SCHEMA.keys()):
      raise ValueError(f"DataFrame schema does not match UNIFIED_SCHEMA. Expected keys: {list(UNIFIED_SCHEMA.keys())}, Got: {list(df.schema.keys())}")
    for col, expected_dtype in UNIFIED_SCHEMA.items():
      if df[col].dtype != expected_dtype:
        raise ValueError(f"DataFrame column {col} has dtype {df[col].dtype}, expected {expected_dtype}")
    return True