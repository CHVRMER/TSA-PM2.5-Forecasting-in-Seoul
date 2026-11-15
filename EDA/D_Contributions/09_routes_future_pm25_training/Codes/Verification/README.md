# Independent numerical verification

This verifier reproduces numerical EDA results from the existing source data.
It does not fit a prediction model or change source datasets.

## Git Bash execution

Use the same environment as the main analysis (`numpy`, `pandas`, `pyarrow`).
Both paths are parameters; no runtime or user directory is hard-coded in Python.

```bash
TSA_PROJECT='C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project'
TSA_EDA="$TSA_PROJECT/EDA/Geographic_Region_Future_PM25_Training_2021_2022"
python "$TSA_EDA/Codes/Verification/verify_geographic_eda.py" \
  --project-root "$TSA_PROJECT" \
  --eda-root "$TSA_EDA"
```

The output is `Results/independent_statistics_verification.json`. To preserve an
earlier audit JSON, supply `--report-path` with a new destination.

## Exact inputs

Relative to `--project-root`:

- `Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet`: authoritative
  training origin/target timestamps, eligibility, future observed PM2.5, season.
- `Dataset/Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet`:
  independent preexisting current PM2.5. No persistence-filled value is used.
- `Dataset/Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet`:
  exact station `111121`, valid quality flag, cross-check of all current values.

Relative to `--eda-root`:

- `Results/origin_visits.parquet`: geographic segment passage indicators.
- `Results/analysis_configuration.json`: fixed bootstrap settings.
- `Results/passage_rates.csv`, `raw_region_future_outcomes.csv`,
  `adjusted_region_contrasts.csv`: results being checked.
- `Codes/run_geographic_eda.py`: only bootstrap sampling and optimized summation
  functions are imported for direct comparison against explicit sampled indices.

## Checks and scope

- Exactly 5,840 training origins, 20 regions, three historical trajectory windows.
- Targets at 3/6/12/24 hours; no target crossing into 2023 is eligible.
- Training eligible target counts: 5,762 / 5,761 / 5,759 / 5,755.
- Exact current concentration has 77 missing origins, left unfilled.
- Current bins are `[0,15)`, `[15,35)`, `[35,75)`, `[75,infinity)`.
- High PM is hourly concentration `>=35`, a research category.
- Raw statistics keep eligible outcomes with unknown current concentration.
- Adjusted contrasts exclude unknown current concentration, use season × current
  bins with at least 10 passage and 10 nonpassage outcomes, and weight by the
  original pooled population on each region's common support.
- Recomputes 60 passage-rate rows, 480 passage/nonpassage raw rows and 240 adjusted
  rows using separate pandas group calculations.
- Checks calendar block sums against explicit sampled row indices with missing
  slots kept in place, and checks the 80% valid-replicate threshold.
- Independently recomputes all 500 bootstrap contrasts and confidence intervals
  for Gangwon and Shandong, window 07–48 hours, forecast horizon 6 hours.

This numerical audit assumes the passage classifications provided by the
geographic classifier. Its separate city/boundary tests cover polygon assignment.
Matching numbers do not establish causality, pollution-source contributions, or
comparability of adjusted estimates using different common-support populations.
