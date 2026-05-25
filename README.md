# PII Curation Pipeline

This project builds a production-style data curation pipeline for AI training data.
It ingests raw records from three formats (Slack JSON exports, CSV tables, and plain-text comment blocks),
detects personally-identifiable information (PII) using a two-layer approach (compiled regex + spaCy NER),
anonymizes every span with hash-based pseudonymization or typed redaction tokens,
scores each record across four quality dimensions with vectorized NumPy operations,
and emits a full per-record audit trail as a JSONL lineage log alongside a Parquet output ready for model training.

---

## Architecture

```
Raw Data (JSON / CSV / TXT)
         |
         v
 IngestOrchestrator
 (format-specific adapters)
         |
         v
  Unified DataFrame
  [record_id, source_format, source_file,
   timestamp, text, raw_metadata,
   source_lineage_tag]
         |
         v
    PIIDetector
  (regex layer + spaCy NER layer)
         |
         v
    PII Detections
  [pii_type, matched_text, start, end,
   confidence, layer]
         |
         v
     Anonymizer
  (pseudonymization + redaction)
         |
         v
  Anonymized Text
  [anonymized_text, anonymization_log,
   anonymization_count, strategies_used]
         |
         v
   QualityScorer
  (NumPy, 4 dimensions)
         |
         v
  Quality Scores
  [completeness_score, consistency_score,
   format_conformance_score,
   pii_removal_confidence_score,
   quality_score]
         |
         v
   LineageLogger
  (JSONL audit log)
         |
         v
data/processed/curated_dataset.parquet
data/lineage/lineage_log.jsonl
data/quality_reports/quality_report.json
```

---

## Project Layout

```
pii-curation-pipeline/
|-- requirements.txt
|-- generate_data.py          # one-time synthetic data generator (Faker, seed 42)
|-- pipeline/
|   |-- ingest.py             # DataAdapter ABC + SlackAdapter, CSVAdapter, PlainTextAdapter
|   |-- pii_detector.py       # regex layer + spaCy NER + overlap deduplication
|   |-- anonymizer.py         # hash pseudonymization + typed redaction
|   |-- quality_scorer.py     # vectorized NumPy 4-dimension scorer + JSON report
|   |-- lineage_logger.py     # JSONL lineage log + summary sidecar
|   `-- main.py               # end-to-end pipeline orchestrator
|-- data/
|   |-- raw/
|   |   |-- slack_messages.json   (200 messages, 30% PII)
|   |   |-- project_records.csv  (300 rows, 40% PII in notes)
|   |   `-- code_comments.txt    (150 blocks, 25% PII)
|   |-- processed/               # curated_dataset.parquet written here
|   |-- lineage/                 # lineage_log.jsonl + summary sidecar written here
|   `-- quality_reports/         # quality_report.json written here
|-- notebooks/
|   `-- eda_quality_analysis.ipynb
`-- tests/
    |-- test_pii_detector.py     (37 tests)
    |-- test_anonymizer.py       (41 tests)
    |-- test_quality_scorer.py   (50 tests)
    `-- test_lineage_logger.py   (61 tests)
```

---

## Prerequisites

- Python 3.11 or later

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> `hashlib` is part of the Python standard library and is not listed in `requirements.txt`.

---

## Generate Synthetic Raw Data

Run once to create the three raw data files under `data/raw/`:

```bash
python generate_data.py
```

---

## Run Full Pipeline

Runs all seven stages end-to-end and writes outputs to `data/`:

```bash
python -m pipeline.main
```

Expected output (approx. 10 seconds on a laptop):

```
[1/7] Ingesting data sources ...
      644 records loaded from 3 source(s).
[2/7] Detecting PII (regex + spaCy NER) ...
      PII found in 527 of 644 records.
[3/7] Anonymizing PII ...
      948 PII spans replaced across all records.
[4/7] Scoring record quality ...
      642/644 records passed threshold 0.8 (mean score: 0.986).
[5/7] Logging lineage events ...
      2880 events flushed to: data/lineage/lineage_log.jsonl
[6/7] Generating quality report ...
      Report saved to: data/quality_reports/quality_report.json
[7/7] Saving processed dataset to Parquet ...
      Dataset saved to: data/processed/curated_dataset.parquet

--------------------------------------------------------------
  Total records ingested:            644
  PII detected in:                   527 records
  Records passed quality threshold:  642
  Quality report saved to:           data/quality_reports/quality_report.json
  Lineage log saved to:              data/lineage/lineage_log.jsonl
  Dataset saved to:                  data/processed/curated_dataset.parquet
  Elapsed:                           9.72s
--------------------------------------------------------------
```

---

## Run Ingestion Only

Loads all three sources and prints a summary without running detection or scoring:

```bash
python -m pipeline.ingest
```

---

## Run EDA Notebook

```bash
jupyter notebook notebooks/eda_quality_analysis.ipynb
```

The notebook loads `data/processed/curated_dataset.parquet` and produces:

- Dataset overview with null counts and column types
- PII distribution by source format
- Quality score distribution (histogram + box plot)
- Dimension score heatmap (4 dimensions x 3 formats)
- Low-quality segment flagging
- PII anomaly scatter plot
- Summary findings and next-step recommendations

---

## Run Tests

```bash
pytest tests/ -v
```

All 189 tests pass with no external dependencies beyond the installed packages.
Tests use `scope="module"` fixtures so spaCy loads once per file, keeping the full suite under 10 seconds.

---

## Interview Demo Path

Run these four commands in order to walk through the complete project:

```bash
python -m pipeline.ingest
python -m pipeline.main
pytest tests/ -v
jupyter notebook notebooks/eda_quality_analysis.ipynb
```

---

## Design Decisions

### Why use both regex and spaCy NER?

Regex is fast and precise for structured patterns with known formats — emails, phone numbers,
SSNs, credit card numbers, and IP addresses all follow strict syntactic rules where a well-written
pattern produces near-zero false negatives. spaCy's NER model handles contextual entities —
person names, organizations, and geographic locations — that have no reliable structural pattern
and require understanding the surrounding sentence. Using both layers in sequence, with a
confidence-ranked deduplication pass to remove overlapping spans, gives higher recall than either
approach alone.

### Why pseudonymize some PII and redact other PII?

Pseudonymization replaces a PII value with a consistent, reversible-by-design token that preserves
referential integrity across records. An email like `alice@corp.com` always becomes `[EMAIL_3fa2c1d0]`
regardless of which record it appears in, so co-reference relationships survive anonymization and
the dataset remains analytically useful. Pseudonymization is applied to identifiers that carry
individual-level signal useful for downstream models: EMAIL, PHONE_US, SSN, CREDIT_CARD, PERSON.

Redaction discards the value entirely, replacing it with a typed placeholder (`[REDACTED_ORG]`).
It is applied to broader categorical entities — organizations, locations, IP addresses — where
the specific value matters less than the fact that a sensitive category was present, and where
preserving the exact token across records could enable re-identification through combination
with other fields.

### Why use NumPy/vectorized scoring?

Scoring is applied to every row in the DataFrame simultaneously using `np.select` and vectorized
pandas string operations (`str.len`, `str.count`, `pd.to_datetime(errors="coerce")`). This avoids
Python-level loops, which would scale poorly on large datasets. The scoring code is also
stateless and side-effect-free, making it straightforward to test with isolated DataFrames.

### Why use JSONL for lineage logging?

Newline-delimited JSON keeps the log human-readable and grep-able without loading the entire
file. Each event is a self-contained JSON object on its own line, so the log can be appended
to across pipeline runs, processed line-by-line in streaming fashion, or imported directly
into any log aggregator (Elasticsearch, Splunk, BigQuery) that supports JSONL ingestion.
A compact summary sidecar is written alongside the JSONL on every flush for quick inspection
without parsing the full file.

### Why keep source lineage tags?

Every record in the unified DataFrame carries a `source_lineage_tag` in the format
`format::filepath::timestamp`. This field drives the `format_conformance_score` dimension
in quality scoring, and it means that any record in the final Parquet output can be traced
back to the exact file and ingestion timestamp it came from. In production data curation,
traceability is as important as the data itself — it enables root-cause analysis when a
model trained on the data behaves unexpectedly.

---

## Output Files

| File | Description |
|---|---|
| `data/processed/curated_dataset.parquet` | 644 records x 21 columns; all complex columns serialized as JSON strings for Parquet compatibility |
| `data/lineage/lineage_log.jsonl` | 2,880 events: 644 ingested + 644 pii_detected + 948 anonymized + 644 quality_scored |
| `data/lineage/lineage_log_summary.json` | Event counts by type, unique record count, run ID, log file path |
| `data/quality_reports/quality_report.json` | Mean / median / std quality score, % above threshold, PII stats, dimension weights |

---

## Dependencies

| Package | Purpose |
|---|---|
| pandas | DataFrame I/O and manipulation |
| numpy | Vectorized quality scoring arithmetic |
| spacy | Named-entity recognition (NER) for PERSON, ORG, GPE |
| faker | Synthetic PII generation for raw data files |
| matplotlib | Chart rendering in EDA notebook |
| seaborn | Statistical chart styling in EDA notebook |
| pyarrow | Parquet read/write for processed output |
| pytest | Test runner |
| jupyter | Notebook server |
