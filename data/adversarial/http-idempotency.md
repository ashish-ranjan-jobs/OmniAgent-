# Idempotency in HTTP

Safe methods (GET, HEAD) and idempotent methods (PUT, DELETE) can be retried without duplicating effects: the server state after one request equals the state after several. POST is neither, so payment APIs attach client-generated idempotency keys: the server records the key with the first response and replays that response for duplicate submissions instead of executing twice. Retry storms after timeouts are the classic motivator.
