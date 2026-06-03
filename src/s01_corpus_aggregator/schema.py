import polars as pl

UNIFIED_SCHEMA = {
  "recipe_id": pl.Utf8,
  "source_dataset": pl.Utf8,
  "language": pl.Utf8,
  "raw_ingredients": pl.List(pl.Utf8)
}