@echo off
REM ERA5 for the CorrDiff spike (tasks/spike_corrdiff.md): 2026-05 .. 2026-09 over a
REM wider window, into data\raw\era5\era5_corrdiff_2026.zarr. Resumable: months
REM already in the store are skipped, so re-run after an interruption.
cd /d "%~dp0.."
if not defined PY set PY=python
"%PY%" -W ignore -X utf8 scripts\download_era5.py --year 2026 --months 5 6 7 8 9 --area 3.0 102.0 -0.5 105.5 --store era5_corrdiff >> logs\era5_corrdiff.log 2>&1
echo === era5_corrdiff exited %DATE% %TIME% === >> logs\era5_corrdiff.log
