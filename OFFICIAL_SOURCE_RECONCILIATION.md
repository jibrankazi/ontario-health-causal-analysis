# Ontario PHU publisher comparison

The new `scripts/official_phu_crosscheck.py` fetches a real Ontario Ministry of
Health resource from the official Ontario Data Catalogue:
https://data.ontario.ca/dataset/status-of-covid-19-cases-in-ontario-by-public-health-unit-phu

The official resource has **daily PHU status snapshots** (active, resolved and
deceased case counts), keyed by `FILE_DATE` and `PHU_NUM`. The repository
holds **weekly incidence** with an undocumented derivation and treatment flags.
They are not the same statistical measure. We compare actual official source
identity codes and dates and record the publisher file SHA-256 and coverage,
but NEVER assert that the original weekly incidence values match the official
daily status counts. If the official server is unavailable, the resulting
report transparently records an unverified retrieval, not a fabricated pass.

Run `python -m scripts.official_phu_crosscheck`, then inspect
`results/official_phu_comparison.json`. GitHub Actions uploads the same
machine-readable verification attempt.

A causal study requires a documented **specific policy** with true variation
in PHU exposure, dated rollout, comparable units, a reproducible outcome
definition with population denominators, proper pre-trend and few-treated-
cluster inference, and verified source-level provenance. A province-wide
policy cannot turn the 29 PHUs labeled 'comparison' into untreated controls
by assertion. The project does not establish a causal policy effect.
