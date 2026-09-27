@echo off
REM One-off: backfill ~31 days of the NEA 240 km radar (NEA keeps ~30 days).
REM Safe to re-run: existing images are skipped.
cd /d "%~dp0.."
python -X utf8 scripts\scrape_radar.py --product 240km --hours 744 > logs\backfill_240km.log 2>&1
echo === backfill exited %DATE% %TIME% === >> logs\backfill_240km.log
