# Sharding Vector Databases

Vector collections outgrow one node, so deployments shard by document id or entity hash and replicate shards for availability. Queries fan out to all shards (scatter-gather) and merge top-k lists, or routing tables consult only shards owning the entity. Rebalancing moves vectors plus their index structures, so capacity planning happens early; cross-shard recall loss from per-shard top-k truncation is a real accuracy cliff.
