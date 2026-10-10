# Idempotent Ingestion

Ingestion is idempotent when running it twice over the same corpus produces
no duplicate side effects. The implementation here hashes each chunk's
content (SHA-256) and stores the hash as a unique column: re-ingesting an
unchanged corpus skips every chunk, and editing one document re-embeds only
the chunks whose content actually changed.

This is the storage-side twin of idempotency keys in payment systems: the
same input maps to at most one stored effect, so retries after failures are
safe and cron-driven re-ingestion never bloats the index. It also makes
re-embedding after an embedding-model upgrade an explicit, measured operation
instead of accidental duplication.
