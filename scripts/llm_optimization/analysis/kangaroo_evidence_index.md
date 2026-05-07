# Kangaroo Evidence Index

This file keeps the Kangaroo evidence set readable after removing bulky raw traces,
debug GIFs, and exploratory outer-loop run folders.

Kangaroo is currently exploratory evidence, not a solved canonical game. The compact
evidence retained here supports the main thesis conclusion: LeGPS can discover partial
Kangaroo behaviours, but the current loop does not reliably solve the ladder and
post-200 transition mechanics.

## Retained Evidence

### Route and Mechanics Traces

- `analysis/traces/kangaroo_policy_v6_overlap_patch_route_window_300_380.json`
  - Compact route-window trace for the ladder-overlap issue.
  - Shows the policy reaching the ladder region but triggering ladder behaviour too
    early or unreliably.

- `analysis/traces/kangaroo_shaped_v2_post200_action_probe.json`
  - Local action probe from post-200 states.
  - Shows that the failure is not solved by a single obvious action.

- `analysis/traces/kangaroo_top_ladder_transition_probe.json`
  - Bounded search over 216 short action-sequence candidates after the first
    200-point reward.
  - No candidate exceeded score 200.

- `analysis/traces/kangaroo_top_ladder_best_prefix_long_trace.json`
  - Long replay of the best short forced prefix.
  - Confirms that the best prefix delays failure but still returns to the
    top-ladder death state at score 200.

### Evaluation Summaries

- `analysis/evaluations/kangaroo/kangaroo_policy_v6_overlap_patch2_retuned_vectorized_128eps_50000steps.json`
  - Main compact evidence that the retuned overlap family can sometimes score.

- `analysis/evaluations/kangaroo/kangaroo_policy_v6_overlap_patch2_vectorized_128eps_50000steps.json`
  - Comparison point before the retuned overlap patch.

- `analysis/evaluations/kangaroo/kangaroo_shaped_v2_post200_guard_eval32_10000steps.json`
  - Shows the guarded post-200 policy still plateauing.

- `analysis/evaluations/kangaroo/kangaroo_shaped_v2_post200_guard_dodge_eval32_10000steps.json`
  - Shows that adding a dodge guard did not solve the post-200 transition.

- `analysis/evaluations/kangaroo/kangaroo_dense_reward_20260505_policy_v3_eval32_10000steps.json`
  - Compact validation of the first policy-provided dense-reward run.
  - Confirms that the dense reward mechanism works technically but did not solve
    Kangaroo.

### Visual Evidence

- `analysis/videos/kangaroo_policy_v6_overlap_patch2_retuned_ep000_success_first1000_debug.gif`
  - Positive visual evidence: the retuned overlap family can produce useful
    early behaviour.

- `analysis/videos/kangaroo_policy_v6_overlap_patch2_retuned_ep006_fail200_debug.gif`
  - Negative visual evidence: the same policy family can still fail at the
    200-point plateau.

- `analysis/videos/kangaroo_shaped_fresh_20260501_policy_v2.gif`
  - Compact shaped-policy visual evidence.

## Removed Evidence

The following Kangaroo artifacts were intentionally removed:

- Full debug traces with per-frame records.
- Duplicate first-900 or first-1000 trace JSONs when a compact route/probe summary
  already captured the conclusion.
- Old smoke, resume, anti-punch, fresh, and shaped-fresh run folders under
  `runs/single_game` and `runs/unified_suite`.
- The full `kangaroo_20260505_154215_dense_reward_s20260505` dense-reward run
  folder, after preserving its result in a compact evaluation summary.
- Redundant debug GIFs for intermediate policies.

Those files were useful during debugging, but they are too large and too detailed
for the thesis evidence layer. Their conclusions are preserved in milestones,
evaluation summaries, retained compact traces, and this index.

## Current Conclusion

The best Kangaroo evidence shows partial progress:

- ladder alignment was identified as a perception/geometry bottleneck;
- retuned overlap logic could sometimes produce reward;
- the policy family was not robust;
- shaped and dense objectives helped create better search signals but did not
  solve the post-200 top-ladder transition;
- further Kangaroo work should be staged mechanics discovery, not another broad
  outer-loop rerun from the same setup.
