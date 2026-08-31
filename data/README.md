# Analytical data access

The analytical workbook is not included in this public repository. The formal
run used an ordered, de-identified table with 1,469 rows, 14 predictors, and
one three-level injury outcome. It contains temporal context, roadway context,
crash interaction, vehicle-history variables, and the injury class.

The source reports are publicly downloadable from the
[NHTSA Standing General Order crash-reporting page](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting).
Users should consult the original documentation for reporting scope, field
definitions, and redistribution terms.

To run the full workflow after obtaining and preparing the data, place the
workbook at `data/ads_crashes.xlsx`. The reported formal run used a workbook
with the following SHA-256 identity:

```text
5d70b42898355269607df61a7ca8b2ac3255193d5cbd846fef9398a7c577c205
```

Aggregate result tables are included under `results/`, so the manuscript
figures can be regenerated without the workbook.
