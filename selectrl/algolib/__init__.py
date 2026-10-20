"""Pure math helpers that algorithms may import.

Modules:
  combin    exact binomials, the unbiased pass@k estimator, capped multi-cluster draw counts and the pass@k
            leave-one-out advantage with its leave-one-out baseline.
  selector  the deployment selector (Chk filter + signature majority vote): class partition of a group, selector
            success on a subset, and exact select@k / cov@k over all k-subsets of a pool.
  sspo_dp   the SSPO dynamic programs (COUNTS / TABLE / EXPECT / J / DIFF / UHAT) for the selection
            leave-one-out advantage, and the full-enumeration fallback for impure clusters.

Everything here is CPU-only, deterministic, and uses exact integer arithmetic for combinatorial counts.
"""
