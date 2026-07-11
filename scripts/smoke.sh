#!/usr/bin/env bash
# Smoke-test the compose infrastructure: healthchecks, llama.cpp, infinity,
# the postgres schema, and elasticsearch. Run after `docker compose up -d --wait`:
#
#   bash scripts/smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Pull the values the checks need (.env is the compose interpolation source).
set -a
# shellcheck disable=SC1091
source .env
set +a

pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

echo "[1/5] Service health states"
for svc in llamacpp infinity-embeddings postgres elasticsearch; do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$svc" 2>/dev/null) \
        || fail "$svc: container not found (is the stack up?)"
    [ "$status" = "healthy" ] || fail "$svc: health is '$status', expected 'healthy'"
    pass "$svc healthy"
done
docker compose ps

echo "[2/5] llama.cpp"
curl -fsS http://localhost:8080/health >/dev/null || fail "llama.cpp /health not OK"
pass "/health OK"
models_json=$(curl -fsS http://localhost:8080/v1/models) || fail "llama.cpp /v1/models failed"
echo "$models_json" | grep -q "$BASE_MODEL" \
    || fail "/v1/models does not list ${BASE_MODEL}: ${models_json}"
pass "/v1/models lists ${BASE_MODEL}"

echo "[3/5] infinity embeddings"
# The host port binding may be interface-specific (e.g. 192.168.86.21:8081),
# so resolve the reachable address instead of assuming localhost.
infinity_addr=$(docker compose port infinity-embeddings 8081) \
    || fail "could not resolve the infinity host port"
curl -fsS "http://${infinity_addr}/health" >/dev/null || fail "infinity /health not OK"
pass "/health OK (${infinity_addr})"
embedding_dim=$(curl -fsS -X POST "http://${infinity_addr}/v1/embeddings" \
    -H "Authorization: Bearer ${secret_infinity_key}" \
    -H 'Content-Type: application/json' \
    -d '{"model": "infloat/multilingual-e5-large-instruct", "input": ["This is a sample sentence!"]}' \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))') \
    || fail "embeddings request failed"
[ "$embedding_dim" = "1024" ] || fail "embedding dimension is ${embedding_dim}, expected 1024"
pass "/v1/embeddings returns a 1024-dim vector"

echo "[4/5] postgres schema"
psql_c() { docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "$1"; }
psql_c "SELECT extname FROM pg_extension" | grep -q vector || fail "'vector' extension missing"
pass "vector extension installed"
psql_c '\d documents' >/dev/null || fail "'documents' table missing"
psql_c '\d chunks' >/dev/null || fail "'chunks' table missing"
pass "documents + chunks tables exist"
indexes=$(psql_c "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
for idx in chunks_embedding_hnsw chunks_doc_id_idx chunks_doc_orig_uidx; do
    echo "$indexes" | grep -q "$idx" || fail "index '$idx' missing on chunks"
done
pass "3 chunk indexes exist (hnsw, doc_id, doc+original_index unique)"

echo "[5/5] elasticsearch"
cluster_status=$(curl -fsS http://localhost:9200/_cluster/health \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])') \
    || fail "elasticsearch /_cluster/health not reachable"
# Single-node clusters are yellow by design (replicas stay unassigned).
case "$cluster_status" in
    yellow|green) pass "cluster health is '${cluster_status}' (yellow is healthy on single-node)" ;;
    *) fail "cluster health is '${cluster_status}', expected yellow or green" ;;
esac

echo
echo "All smoke checks passed."
