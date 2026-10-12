# CanLegalRAGBench ingestion on the lab machine → laptop

How to build the exploration app's OpenSearch index (`canlegal-qwen3`) on the lab GPU
and copy it to the laptop. Written so a Claude Code session on either machine can follow
it step by step. Run the steps in order, and check each **Verify** line before moving on.

## Why the lab

Embedding ~63K chunks with Qwen3-Embedding-0.6B takes **5–9 hours on the laptop's M2**
and **minutes on the lab RTX 4090**. So we build the index on the lab machine, then move
it to the laptop with an OpenSearch snapshot. A snapshot copies vectors and HNSW graphs
exactly, so nothing is re-embedded.

## Machines

| | Lab (`secb1010-d11.csil.sfu.ca`, SSH port 24, user `twa174`) | Laptop (macOS, M2) |
|---|---|---|
| Repo | `~/LegalRAG` | `~/sfu/nlp/LegalRAG` |
| Python | `source ~/LegalRAG/.venv/bin/activate` | `conda activate legalrag` |
| OpenSearch | Installed app at `~/opensearch-3.4.0` (no Docker) | Docker: `docker compose up -d opensearch` |
| Version | 3.4.0 | 3.4.0 (image `opensearchproject/opensearch:3.4.0`) |
| Snapshot dir | `/home/twa174/opensearch-snapshots` | `./opensearch-snapshots` (repo root; mounted into the container at `/usr/share/opensearch/snapshots`) |

## Fixed ingestion settings (do not change without re-deciding)

These were decided on 2026-10-12. They must match the API's `EXPLORE_*` settings in
`legalrag/core/config.py`:

| Setting | Value | Why |
|---|---|---|
| Index name | `canlegal-qwen3` | Default `EXPLORE_INDEX_NAME` |
| Embedder | `sentence_transformers` / `Qwen/Qwen3-Embedding-0.6B` (1024-dim) | Default `EXPLORE_EMBEDDING_MODEL`; queries are embedded the same way (no instruction prompt), as in UBC's code |
| Chunker | Flat `RecursiveCharacterTextSplitter`, 1024 chars, 128 overlap | Same as UBC's reference config (NLP-UBC/CanLegalRAGBench `configs/base.yaml`); LangChain-style whole-piece overlap |
| Corpus | All 1,117 documents of `UBC-VL/CanLegalRAGBench` `documents.json` | The script downloads it to `data/CanLegalRAGBench/` (gitignored) |

Court names and jurisdictions come from `legalrag/ingestion/canlegal_courts.yaml`.
**Jurisdiction codes in that file must stay quoted:** YAML reads a bare `ON` as `true`.
Editing that file means re-ingesting.

## Step 1 — Lab: update code and start OpenSearch

```bash
cd ~/LegalRAG
git pull origin main
source .venv/bin/activate
pip install -e ".[dev]"            # picks up any new deps; the API's [app] extra is not needed for ingestion

curl -s localhost:9200/_cluster/health || (~/opensearch-3.4.0/bin/opensearch > ~/opensearch.log 2>&1 &)
```
**Verify:** `curl -s localhost:9200 | grep '"number"'` shows `3.4.0`, and the cluster health
is `green` or `yellow`.

## Step 2 — Lab: ingest

```bash
# Only if re-ingesting: delete the old index first
curl -X DELETE localhost:9200/canlegal-qwen3

python scripts/ingest_canlegal.py --index-name canlegal-qwen3 \
    --embedding-provider sentence_transformers \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --chunk-size 1024 --chunk-overlap 128
```
**Verify:**
- The script ends with `Done: 1117 documents, ~62,900 chunks → index 'canlegal-qwen3'`.
- `curl localhost:9200/canlegal-qwen3/_count` reports the same chunk count.
- The embedding dimension is 1024:
  ```bash
  curl -s localhost:9200/canlegal-qwen3/_mapping | python3 -c "import json,sys; m=json.load(sys.stdin); print(list(m.values())[0]['mappings']['properties']['embedding']['dimension'])"
  ```

Expected chunk count: the laptop measured **62,942** chunks for this exact config. A
noticeably different number means the code or flags differ from the laptop's.

## Step 3 — Lab: snapshot the index

One-time setup: allow OpenSearch to write snapshots to `~/opensearch-snapshots`, then
restart it.
```bash
mkdir -p ~/opensearch-snapshots
grep -q '^path.repo' ~/opensearch-3.4.0/config/opensearch.yml || \
  echo 'path.repo: ["/home/twa174/opensearch-snapshots"]' >> ~/opensearch-3.4.0/config/opensearch.yml
kill $(pgrep -f opensearch); sleep 5
~/opensearch-3.4.0/bin/opensearch > ~/opensearch.log 2>&1 &
until curl -s localhost:9200 >/dev/null; do sleep 2; done

curl -X PUT localhost:9200/_snapshot/canlegal -H 'Content-Type: application/json' \
     -d '{"type":"fs","settings":{"location":"/home/twa174/opensearch-snapshots"}}'
```
Each time you snapshot (bump the suffix, e.g. `-v2`, for later re-ingests):
```bash
curl -X PUT "localhost:9200/_snapshot/canlegal/canlegal-qwen3-v1?wait_for_completion=true" \
     -H 'Content-Type: application/json' \
     -d '{"indices":"canlegal-qwen3","include_global_state":false}'
```
**Verify:** the response has `"state":"SUCCESS"` and `"failed":0`. Also
`curl localhost:9200/_snapshot/canlegal/_all` lists the snapshot.

## Step 4 — Laptop: copy and restore

The laptop's `docker-compose.yml` already sets `path.repo` and mounts `./opensearch-snapshots`.
```bash
cd ~/sfu/nlp/LegalRAG
docker compose up -d opensearch
rsync -avz secb1010-d11.csil.sfu.ca:opensearch-snapshots/ ./opensearch-snapshots/   # ~/.ssh/config supplies port 24 + user

curl -X PUT localhost:9200/_snapshot/canlegal -H 'Content-Type: application/json' \
     -d '{"type":"fs","settings":{"location":"/usr/share/opensearch/snapshots"}}'
curl -X DELETE localhost:9200/canlegal-qwen3          # restore fails if the index exists
curl -X POST "localhost:9200/_snapshot/canlegal/canlegal-qwen3-v1/_restore?wait_for_completion=true" \
     -H 'Content-Type: application/json' -d '{"indices":"canlegal-qwen3"}'
```
**Verify:** `curl localhost:9200/canlegal-qwen3/_count` matches the lab count. Then start
the API (`uvicorn legalrag.api.app:app --port 8000`); `curl localhost:8000/api/health`
should report the same number of chunks.

The hybrid search pipeline (`legalrag_hybrid_pipeline`) is cluster state, not part of the
snapshot. The API recreates it on startup.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `[canlegal] location [...] doesn't match any of the locations specified by path.repo` | `path.repo` is missing or OpenSearch wasn't restarted after adding it (lab), or the container wasn't recreated after editing `docker-compose.yml` (laptop: `docker compose up -d opensearch`). |
| `Query vector has invalid dimension: 768. Dimension should be: 1024` at query time | The API embedder doesn't match the index. Check `EXPLORE_EMBEDDING_MODEL`. Never set `populate_by_name` on `ExploreSettings`: it makes `.env`'s `EMBEDDING_MODEL` leak in. |
| Restore error: `index with same name already exists` | `curl -X DELETE localhost:9200/canlegal-qwen3` first, or restore with `"rename_pattern"`/`"rename_replacement"`. |
| `AccessDeniedException` writing snapshots (lab) | `~/opensearch-snapshots` must be writable by the user running OpenSearch (`twa174`). |
| Ingest is slow on the lab | Check the GPU is used: `python -c "import torch; print(torch.cuda.is_available())"`, and look at `nvidia-smi` during the run. |
