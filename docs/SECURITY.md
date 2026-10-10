# OmniAgent Security Posture

## Threat model (what we defend against)

| Threat | Mitigation |
|---|---|
| Secrets in version control | `.env` is gitignored; mode 600; DB password generated at setup, never printed or committed |
| Database exposure | Postgres bound to loopback only (`docker-compose.yml` ports); UFW has no DB rule; container on an internal compose network |
| LLM endpoint exposure | Ollama binds loopback (default); no inbound firewall rule |
| API exposure | uvicorn binds `localhost` by default (`API_HOST`); access from outside requires an SSH tunnel; optional `API_KEY` enforces `X-API-Key` on all `/api/v1` routes when set |
| Prompt-driven abuse via ingest path | Ingest accepts a directory path only; path is server-side; corpus content is treated as untrusted data (never executed) |
| Malicious container escape | Non-root user in the app image; standard images only (`pgvector/pgvector:pg16`, `python:3.12-slim`); no privileged containers; memory limits |
| Resource exhaustion | Pydantic length caps on request bodies (query max 2000 chars); docker `mem_limit` on db and app; reflection budget caps LLM spend per request |
| Data leaving the host | None at runtime: no external LLM APIs, no third-party eval services; package/model downloads occur at setup time only |

## Hardening decisions worth explaining in review

- **Fail-closed verification** (fact-checker) is also a security property: an
  unverifiable draft is treated as unsupported, never silently trusted.
- **Least privilege in containers**: the API image drops to a dedicated `app`
  user; the database container runs the stock non-root entrypoint.
- **Loopback-first networking**: every service (API, DB, Ollama) binds to
  loopback by default. Exposure is an explicit, single-line config change,
  documented here rather than done preemptively.
- **Structured logging without payload leakage**: prompts/answers are not
  logged (only lengths, timings, models) to keep query logs low-sensitivity.

## Explicitly NOT covered (accepted risks, current scope)

- No user authentication/authorization model (single-tenant local deployment).
- No rate limiting (a public deployment should add reverse-proxy limits).
- No TLS termination (localhost/SSH-tunnel deployment; nginx on this host is
  unrelated to this project and was not touched).
- In-process BM25 and CPU models are sized for a small corpus; a multi-tenant
  deployment would need per-tenant isolation.
