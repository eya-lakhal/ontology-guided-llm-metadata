"""Central project paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = ROOT / "ontology"
ONTOLOGY_FILE = ONTOLOGY_DIR / "abx_resources.ttl"
SHAPES_FILE = ONTOLOGY_DIR / "shapes.ttl"
EXAMPLES_FILE = ONTOLOGY_DIR / "examples.ttl"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GOLD_DIR = DATA_DIR / "gold"
QUERIES_DIR = ROOT / "queries"
OUTPUT_DIR = ROOT / "output"
