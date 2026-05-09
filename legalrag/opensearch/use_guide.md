# OpenSearch Interaction Guide

Reference for inspecting and interacting with the LegalRAG OpenSearch index via `curl`.
All commands assume OpenSearch is running at `http://localhost:9200`.

---

## Cluster & Index Health

```bash
# Cluster health (green/yellow/red)
curl http://localhost:9200/_cluster/health

# List all indices with doc count, size, status
curl "http://localhost:9200/_cat/indices?v&s=index"

# Doc counts for index
curl http://localhost:9200/your-index-name/_count
```

---

## Schema Inspection

```bash
# Full index mapping (all fields + types)
curl -s http://localhost:9200/your-index-name/_mapping | python3 -m json.tool

# Check embedding dimension (must match EMBEDDING_DIM in .env)
curl -s http://localhost:9200/your-index-name/_mapping \
  | python3 -c "import json,sys; m=json.load(sys.stdin); \
    print(list(m.values())[0]['mappings']['properties']['embedding']['dimension'])"

# Index settings (ef_search, shard count, analyzers, etc.)
curl -s http://localhost:9200/your-index-name/_settings | python3 -m json.tool
```

---

## Peeking at Documents

```bash
# Fetch 2 child chunks (the ones that are embedded and retrieved)
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 2, "query": {"term": {"is_parent": false}}, "_source": {"excludes": ["embedding"]}}' \
  | python3 -m json.tool

# Fetch 1 parent chunk (stored for context expansion, not embedded)
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 1, "query": {"term": {"is_parent": true}}, "_source": {"excludes": ["embedding"]}}' \
  | python3 -m json.tool

# Fetch a doc by citation (exact keyword match)
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 3, "query": {"term": {"citation": "2015 ONSC 7241"}}, "_source": {"excludes": ["embedding"]}}' \
  | python3 -m json.tool

# Fetch a doc by chunk_id (the document _id in OpenSearch)
curl -s "http://localhost:9200/your-index-name/_doc/<chunk_id>" | python3 -m json.tool

# Full-text search (BM25) on the text field
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 5, "query": {"match": {"text": "standard of review Charter"}}, "_source": {"excludes": ["embedding"]}}' \
  | python3 -m json.tool
```

---

## Sanity Checks

```bash
# Parent/child chunk ratio
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 0, "aggs": {"by_type": {"terms": {"field": "is_parent"}}}}' \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
for b in r['aggregations']['by_type']['buckets']:
    print('is_parent=' + str(b['key_as_string']), '->', b['doc_count'], 'docs')
"

# All unique courts indexed
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 0, "aggs": {"courts": {"terms": {"field": "court", "size": 50}}}}' \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
for b in r['aggregations']['courts']['buckets']:
    print(b['doc_count'], '\t', b['key'])
"

# Date range of indexed decisions
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 0, "aggs": {"min_date": {"min": {"field": "decision_date"}}, "max_date": {"max": {"field": "decision_date"}}}}' \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
a = r['aggregations']
print('earliest:', a['min_date']['value_as_string'])
print('latest:  ', a['max_date']['value_as_string'])
"

# Number of unique documents (distinct doc_ids)
curl -s "http://localhost:9200/your-index-name/_search" \
  -H 'Content-Type: application/json' \
  -d '{"size": 0, "aggs": {"unique_docs": {"cardinality": {"field": "doc_id"}}}}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print('unique docs:', r['aggregations']['unique_docs']['value'])"
```


## Index Lifecycle

```bash
# Delete an index (required when changing EMBEDDING_MODEL / EMBEDDING_DIM)
curl -XDELETE http://localhost:9200/your-index-name

# Re-ingest after deleting (see CLAUDE.md for full ingest commands)
python -m evaluation.LegalBenchRAG.ingest --data-dir data/LegalBenchRAG
python scripts/ingest.py data/subset/

# Refresh index (force flush so newly indexed docs appear in search immediately)
curl -XPOST http://localhost:9200/your-index-name/_refresh
```

---

## Index Schema (for reference)

| Field | Type | Notes |
|---|---|---|
| `text` | `text` | `legal_analyzer`: standard + lowercase + stop words |
| `embedding` | `knn_vector` (384-dim default) | HNSW, cosine, lucene engine; `ef_construction=128`, `m=16`; `ef_search=512` |
| `court`, `citation`, `source_path` | `keyword` | Exact-match filterable |
| `decision_date` | `date` | ISO `yyyy-MM-dd`; range-filterable |
| `is_parent` | `boolean` | `false` = child chunk (embedded); `true` = parent chunk (context only) |
| `chunk_id`, `doc_id`, `parent_chunk_id` | `keyword` | Linkage fields |
| `char_start`, `char_end` | `integer` | Character offsets within the original document |

The `embedding` field is **not returned by default** in search results — exclude it explicitly
with `"_source": {"excludes": ["embedding"]}` to keep output readable.