import time
from pathlib import Path
import polars as pl

from src.s01_corpus_aggregator.parsers.recipenlg import RecipeNLGParser

def main():
  print("Start Stage 1: Corpus Aggregation")
  start_time = time.time()
  
  input_path = "data/s01_raw/recipenlg.csv"
  output_path = "data/s03_processed/unified_corpus.parquet"
  
  Path("data/s03_processed").mkdir(parents=True, exist_ok=True)
  
  parser = RecipeNLGParser(batch_size=250_000)
  
  try:
    df_recipenlg = parser.parse(input_path)
    
    print(f"\n Saving {df_recipenlg.height:,} recipes to {output_path}...")
    df_recipenlg.write_parquet(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"\n Completed in {elapsed_time:.2f} seconds")
  
  except FileNotFoundError:
    print(f"\n Error: Input file not found at {input_path}")
  except Exception as e:
    print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
  main()
    