# Two-computer execution

Both computers must use the same repository revision, analytical workbook,
Python version, and pinned package versions. Do not edit model, seed, fold,
shadow, or explainer settings after either worker starts.

## 1. Prepare both computers

Copy or clone the repository to each computer, then create the environment
shown in the main README. Verify both installations:

```powershell
python scripts/verify_environment.py
python scripts/verify_release.py
```

The workbook hash and all package-version checks must pass on both machines.

## 2. Run the first worker

Assign outer partitions 1, 2, and 3 to the first computer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_predictive.ps1 `
  -Mode worker -WorkerId pc1 -Partitions 1,2,3 -Workers 6 -Pending 16
```

Record the timestamped output directory printed when the worker starts. This is
the aggregation target.

## 3. Run the second worker

Assign outer partitions 4 and 5 to the second computer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_predictive.ps1 `
  -Mode worker -WorkerId pc2 -Partitions 4,5 -Workers 6 -Pending 16
```

When it finishes, copy its entire timestamped output directory to the first
computer. Keep the directory intact.

## 4. Merge the second shard

On the first computer, merge the copied worker output into the PC1 run:

```powershell
python analysis/merge_distributed_checkpoints.py `
  --source incoming_pc2_worker_output `
  --target outputs/<pc1-run-directory>
```

The merger checks dataset identity, statistical fingerprints, completed worker
status, checkpoint markers, and file hashes. It refuses divergent files rather
than overwriting them.

## 5. Aggregate the completed run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_predictive.ps1 `
  -Mode aggregate `
  -ResumeRunDir outputs/<pc1-run-directory> `
  -Workers 6 -Pending 16
```

Aggregation loads the five completed outer partitions and writes the final
tables, figures, and run metadata. Do not delete worker directories until the
output manifest and merge audit have been checked.

## 6. Run the PAG extension

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pag.ps1 `
  -SourceRun outputs/<pc1-run-directory> -Workers 6
```

The PAG workflow can run on either computer after the predictive aggregation is
complete.

