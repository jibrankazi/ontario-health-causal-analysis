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


## Verified February 2021 policy timing (separate from outcome validation)

The [Ontario government's February 8 announcement](https://news.ontario.ca/en/release/60261/ontario-extending-stay-at-home-order-across-most-of-the-province-to-save-lives)
records Hastings and Prince Edward, Kingston-Frontenac-Lennox & Addington,
and Renfrew returning to the framework **February 10, 2021**.
The [February 12 announcement](https://news.ontario.ca/en/release/60339/ontario-returning-27-public-health-regions-to-strengthened-covid-19-response-framework)
documents 27 further PHUs transitioning **February 16**, while Toronto,
Peel, York, and North Bay Parry Sound remained subject to shutdown.
The source comparison now maps these seven specifically named PHUs to the
**independently downloaded official PHU code/name records** and assigns the
remaining 27. It compares those cohorts with the five unsupported
`treated` flags in the checked-in source and the old February 1 cutoff.

This is a verified *policy chronology*, NOT causal identification: early
lifting was selected based on improving local health indicators. The
early-versus-later contrast lasts only six days before other jurisdictions
reopen, shorter than the checked-in weekly outcome cadence. A genuine
causal study therefore needs different daily outcomes, documented covariates,
and a justified design that addresses endogenous region selection.
