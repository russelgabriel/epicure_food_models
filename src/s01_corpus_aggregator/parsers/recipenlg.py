import ast
import polars as pl
from tqdm import tqdm
from ..base_parser import BaseParser

class RecipeNLGParser(BaseParser):
  def __init__(self, batch_size: int = 100_000):
    self.batch_size = batch_size
    
  @staticmethod
  def _safe_parse_list(val: str) -> list[str]:
    """
    Safely parse a stringified list into a list of strings.
    """
    if not val:
      return []
    try:
      return ast.literal_eval(val)
    except Exception:
      return []
      
  def parse(self, input_path: str) -> pl.DataFrame:
    """
    Parses the RecipeNLG dataset and returns a polars DataFrame
    adhering to the UNIFIED_SCHEMA.
    """
    
    # Initialize batch reader
    reader = pl.read_csv_batched(input_path, batch_size=self.batch_size)
    batches = reader.next_batches(10) # Grabs initial chunks
    
    processed_chunks = []
    global_row_offest = 0
    
    total_rows = pl.scan_csv(input_path).select(pl.len()).collect().item()
    pbar = tqdm(total = total_rows, desc="Processing RecipeNLG dataset", unit="rows")
    
    while batches:
      for df_chunk in batches:
        chunk_len = len(df_chunk)
        
        transformed_chunk = (
          df_chunk.with_columns([
            pl.col("NER")
            .map_elements(self._safe_parse_list, return_dtype=pl.List(pl.Utf8))
            .alias("raw_ingredients")
          ])
          .with_columns([
            pl.lit("recipenlg").alias("source_dataset"),
            pl.lit("en").alias("language"),
            (pl.lit("recipenlg_") + pl.arange(global_row_offest, global_row_offest + chunk_len).cast(pl.Utf8)).alias("recipe_id")
          ])
          .select([
            "recipe_id",
            "source_dataset",
            "language",
            "raw_ingredients",
          ])
        )
        
        processed_chunks.append(transformed_chunk)
        global_row_offest += chunk_len
        
        pbar.update(chunk_len)
        
      batches = reader.next_batches(10)
      print(f"Processed {global_row_offest} rows")
    
    pbar.close()
    
    final_df = pl.concat(processed_chunks)
    self.validate_schema(final_df)
    print("Successfully validated RecipeNLG schema layout")
    return final_df