#!/usr/bin/env bash
# Smoke-test the compose infrastructure: healthchecks, llama.cpp, infinity,
# the postgres schema, elasticsearch, prefect, the API, the web app, and the
# observability pair. Run after `docker compose up -d --wait`:
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

echo "[1/10] Service health states"
for svc in llamacpp infinity-embeddings postgres elasticsearch prefect \
           varagity-api varagity-web prometheus grafana; do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$svc" 2>/dev/null) \
        || fail "$svc: container not found (is the stack up?)"
    [ "$status" = "healthy" ] || fail "$svc: health is '$status', expected 'healthy'"
    pass "$svc healthy"
done
docker compose ps

echo "[2/10] llama.cpp"
curl -fsS http://localhost:8080/health >/dev/null || fail "llama.cpp /health not OK"
pass "/health OK"
models_json=$(curl -fsS http://localhost:8080/v1/models) || fail "llama.cpp /v1/models failed"
echo "$models_json" | grep -q "$BASE_MODEL" \
    || fail "/v1/models does not list ${BASE_MODEL}: ${models_json}"
pass "/v1/models lists ${BASE_MODEL}"

echo "[3/10] infinity embeddings"
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

echo "[4/10] postgres schema"
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
# The v2 conversation/settings tables arrive via the migration runner on API
# startup (not schema.sql), so their presence also proves the runner ran.
for tbl in conversations messages message_sources app_settings schema_migrations; do
    psql_c "\\d ${tbl}" >/dev/null || fail "'${tbl}' table missing (migration runner)"
done
pass "migration-runner tables exist (conversations, messages, message_sources, app_settings, schema_migrations)"

echo "[5/10] elasticsearch"
cluster_status=$(curl -fsS http://localhost:9200/_cluster/health \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])') \
    || fail "elasticsearch /_cluster/health not reachable"
# Single-node clusters are yellow by design (replicas stay unassigned).
case "$cluster_status" in
    yellow|green) pass "cluster health is '${cluster_status}' (yellow is healthy on single-node)" ;;
    *) fail "cluster health is '${cluster_status}', expected yellow or green" ;;
esac

echo "[6/10] prefect"
curl -fsS http://localhost:4200/api/health >/dev/null || fail "prefect /api/health not OK"
pass "/api/health OK (UI at http://localhost:4200)"

echo "[7/10] api"
health_json=$(curl -fsS "http://localhost:${API_PORT:-8000}/api/health") \
    || fail "api /api/health not reachable"
echo "$health_json" | python3 -c '
import json, sys
services = json.load(sys.stdin)["services"]
down = [name for name, state in services.items() if not state["ok"]]
sys.exit(1 if down else 0)
' || fail "api reports unreachable dependencies: ${health_json}"
pass "/api/health reports all dependencies reachable"
curl -fsS "http://localhost:${API_PORT:-8000}/openapi.json" >/dev/null \
    || fail "api /openapi.json not OK"
pass "/openapi.json served"
curl -fsS "http://localhost:${API_PORT:-8000}/metrics" | grep -q '^varagity_' \
    || fail "api /metrics missing varagity_* families (METRICS_ENABLED=false?)"
pass "/metrics exposes the varagity_* catalog"

echo "[8/10] web"
curl -fsS http://localhost:3000 >/dev/null || fail "web :3000 not OK"
pass "web served at http://localhost:3000"

echo "[9/10] prometheus"
prom_port="${PROMETHEUS_PORT:-9090}"
curl -fsS "http://localhost:${prom_port}/-/healthy" >/dev/null \
    || fail "prometheus /-/healthy not OK"
pass "/-/healthy OK"
api_target_up=$(curl -fsS "http://localhost:${prom_port}/api/v1/targets" \
    | python3 -c '
import json, sys
targets = json.load(sys.stdin)["data"]["activeTargets"]
print(any(t["labels"].get("job") == "varagity-api" and t["health"] == "up" for t in targets))
') || fail "prometheus targets API failed"
[ "$api_target_up" = "True" ] || fail "the varagity-api scrape target is not up"
pass "varagity-api scrape target is up"

echo "[10/10] grafana"
graf_port="${GRAFANA_PORT:-3001}"
curl -fsS "http://localhost:${graf_port}/api/health" >/dev/null \
    || fail "grafana /api/health not OK"
pass "/api/health OK"
# Anonymous Viewer is provisioned, so the search API works without auth.
dashboards=$(curl -fsS "http://localhost:${graf_port}/api/search?type=dash-db") \
    || fail "grafana search API failed"
for uid in varagity-query varagity-ingestion varagity-infra; do
    echo "$dashboards" | grep -q "$uid" || fail "provisioned dashboard '$uid' missing"
done
pass "3 provisioned dashboards present (query, ingestion, infra) at http://localhost:${graf_port}"

echo
echo "All smoke checks passed."
