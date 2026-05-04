"""Retuned parameters for policy_v6_overlap_patch2.

The policy logic is identical to patch 2. Only the numeric parameters are
replaced by the local CMA-ES result from
kangaroo_policy_v6_overlap_patch2_retune_cma6_eval32_10000steps.json.
"""

import jax.numpy as jnp

from scripts.llm_optimization.experiments.kangaroo.policy_v6_overlap_patch2 import (
    measure_main,
    policy,
)


def init_params():
    return {
        "align_tol": jnp.array(6.1342549324035645),
        "danger_r": jnp.array(18.066486358642578),
        "height_weight": jnp.array(0.6482160091400146),
        "ladder_bottom_slack": jnp.array(3.2354776859283447),
        "ladder_center_tol": jnp.array(3.1794726848602295),
        "min_ladder_overlap_frac": jnp.array(0.21243366599082947),
        "punch_dx": jnp.array(19.652551651000977),
        "reach_tol": jnp.array(15.936382293701172),
        "reach_up": jnp.array(28.66370964050293),
        "route_exclude_x_tol": jnp.array(14.426636695861816),
    }
