# Runtime snapshot — 2026-10-25

This snapshot publishes the installed PromptRefiner Python implementation and its accompanying regression tests. It does not modify MiniMax H3 Director, model weights, personal presets, or local workflows.

Changes include BEAT-embedded dialogue extraction, speaker-only landmark filtering, delivery-direction normalization, shot-contract dialogue preservation, strict segment duration caps, and duration-overrun audit details.

## Validation

Command: `python -m unittest discover -s tests`

- Base main (3adf7af): 121 tests; 1 failure.
- Runtime snapshot: 125 tests; 3 failures and 1 error (121 passing).
- No GPU generation validation was performed for this publication.

Outstanding tests:

- `test_full_pipeline_local_repair_records_raw_and_keeps_packing_contract`: also fails on base main; expects a locked schedule instruction absent from current prompt.
- `test_schedule_does_not_make_an_under_four_second_tail`: expects a seven-second segment with a four-second configured cap; runtime rejects it.
- `test_explicit_final_shots_are_losslessly_bound_to_cards`: expects empty dialogue fields; runtime now extracts dialogue into these fields.
- `test_independent_refine_keeps_each_beat_focused`: expects no duration rebalance; runtime reports rebalance.

These discrepancies require review, not disabled assertions. This branch is a reproducible runtime snapshot, not a fully passing release.
