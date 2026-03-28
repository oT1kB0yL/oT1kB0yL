# Doc Pipeline

A file processing pipeline that extracts structured data from PDF documents (e.g. financial statements, payslips) using OCR and Vision Language Models (VLM).

## Pipeline Overview

```
PDF File  →  OCR with keyword filtering  →  VLM Extraction  →  Normalization  →  JSON Output
```

1. OCR and Page Filtering — scans all pages, applies keyword pre-filtering, then scores and ranks pages using BM25 + vector similarity to select the most relevant ones.
2. **VLM Extraction** — sends images to a VLM to extract structured fields (e.g. revenue, net profit).
3. **Normalization** — cleans, translates, and merges results into a per-year JSON record.

## Supported Document Types

| doc_type | Description |
|---|---|
| `financial_statement` | Annual/quarterly financial statements (revenue, net profit, equity, dividend, etc.) |
| `payslip` | Employee payslips (net pay, deductions, employment, tax payable, etc.) |
| ... | ... |

## Page Retrieval Algorithm

Pages are selected through a two-stage process:

1. **Keyword pre-filter** — hard AND/OR match on predefined keyword groups per page type (`main`, `dividend`, `equity`, `company_name`). Pages that pass are candidates; pages that fail are dropped.
2. **Scoring & ranking** — every page (passed or not) is scored with a weighted combination of:
   - **Keyword hit score** (weight 0.4) — count of keyword groups fully matched
   - **BM25** (weight 0.3) — classical IR relevance score; uses per-document term frequency and inverse document frequency
   - **Vector similarity** (weight 0.3) — cosine similarity between sentence embeddings from the page text and a keyword-reference embedding; captures semantic resemblance beyond exact token overlap
3. **Top-k selection** — the top `page_score_top_k` pages by combined score are kept, merged with the keyword-passed set.

The algorithm gracefully handles cases where no keyword match occurs but the page content is still semantically related to the target document type.

### Tunable Parameters

All parameters live in `src/const/page_retrieval_config.py`:

| Parameter | Default | Description |
|---|---|---|
| `bm25_k1` | 1.5 | BM25 term frequency saturation |
| `bm25_b` | 0.75 | BM25 document length normalisation |
| `keyword_weight` | 0.4 | Weight of keyword hit score in final ranking |
| `bm25_weight` | 0.3 | Weight of BM25 score in final ranking |
| `vector_weight` | 0.3 | Weight of sentence embedding cosine similarity in final ranking |
| `page_score_top_k` | 3 | Number of top-scoring pages to always include |

## Project Structure

```
doc_pipeline/
├── data/                          # Sample PDFs
│   ├── sample-financial-statements.pdf
│   └── sample_termsheet.pdf
├── src/
│   ├── const/
│   │   ├── api_config.py            # API keys & endpoints
│   │   ├── keywords_map.py           # Page retrieval keywords & prompt templates
│   │   ├── page_retrieval_config.py # BM25 / vector similarity parameters
│   │   └── template.py               # VLM prompt templates
│   └── utils/
│       ├── central_service.py        # OCR & VLM inferencer wrappers
│       ├── image_operation.py        # PIL image utilities
│       ├── pdf_operation.py          # Page finding & extraction logic
│       ├── post_processing.py        # Field normalization
│       ├── text_operation.py         # Language detection & OCR helpers
│       └── token_retrieval.py        # Keyword matching, BM25, vector similarity
├── main.py                        # Entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `api_config.py` (or set environment variables) and fill in your keys:

```bash
export OCR_API_KEY="your-key"
export OCR_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VLM_API_KEY="your-key"
export VLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## Usage

```bash
python doc_pipeline/main.py \
    --pdf doc_pipeline/data/sample-financial-statements.pdf \
    --doc-type financial_statement
```

By default the script runs `sample-financial-statements.pdf` with `financial_statement` as the doc type.

## Output

```json
{
  "pdf_path": "...",
  "doc_type": "financial_statement",
  "origin_lang": "EN",
  "data": [
    {
      "year": "2023",
      "revenue": "...",
      "net_profit": "..."
    }
  ]
}
```
