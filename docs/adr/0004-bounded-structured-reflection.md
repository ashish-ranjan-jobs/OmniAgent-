# ADR-0004: Bounded, structured reflection instead of free-form retry

Date: 2026-10-10 | Status: accepted

## Context
A fact-checker that can fail a draft is only safe if failure leads somewhere
bounded. Free-form "try again" loops are non-deterministic, unbounded, and
waste latency.

## Decision
1. The cyclical edge (fact_checker -> planner) is conditional: taken only if
   some claim failed AND reflection_count < MAX_REFLECTIONS (default 2).
2. Feedback is structured, not prose: failed claims + derived suggested
   subqueries ("evidence for the claim: X"), built by a pure, tested function.
3. The planner prompt embeds the failed claims; the fact-checker is strict and
   fail-closed (unparseable verdicts = unsupported).
4. Termination is structural: reflection_count is monotone; at most
   1 + MAX_REFLECTIONS rounds, under LangGraph's recursion limit.

## Consequences
+ Termination provable; latency bounded (each round costs one plan + synth +
   verify cycle plus retrieval).
+ Reflection is measurable (reflection_count distribution, groundedness delta
   with reflection on/off -- an ablation arm in evals).
- A stubborn claim burns the budget and ships flagged rather than silently
  retried forever; that is intentional honesty.
