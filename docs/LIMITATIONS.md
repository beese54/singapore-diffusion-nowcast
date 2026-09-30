# Limitations

Read these before using any result from this repository.

## What the model cannot do yet

1. **Heavy rain is not predicted better than persistence at any lead.** Pooled FSS at ≥10 mm/hr,
   11.9 km: 0.34 vs 0.45 (30 min), 0.06 vs 0.24 (60), 0.09 vs 0.16 (90). The ensemble under-forecasts
   heavy-rain area by about half and its intensity by 5–10× in the case studies. Heavy rain is what
   drives flash floods, so **this is not an operational flood-warning system.**
2. **It cannot see storms that are not on the radar yet.** Input is 30 min of a 35 × 63 km radar
   domain. Cells that form in place, or arrive from outside the domain, are invisible until they
   appear. On 22 Sep the first minutes of a rapidly growing cell were missed; on 30 Sep a storm that
   grew within a few km of two flood sites was flagged only 10–15 min before the rain. Radar-only fixes,
   the 240 km wide-range radar and Himawari satellite were all tried and did not help as tried (see
   [RESULTS.md §6](RESULTS.md)).
3. **Longer leads are weak.** The 60/90-min models only tie or slightly beat persistence for light
   rain, and give no useful heavy-rain signal.
4. **Probabilities are under-confident.** In the 22 Sep case, points given 25–50% were wet 88% of the
   time. A low, non-zero probability near a flood-prone location should be read as a real warning.

## What the evidence can and cannot support

5. **Short archive.** Models were trained on 103 days (22 May – 2 Sep 2026): the Southwest Monsoon and
   inter-monsoon only. The Northeast Monsoon (Dec–Mar), with its long-lived monsoon surges, has never been
   seen. **Every negative result at 60–90 min rests on this one season** and is data-limited; collection
   continues and the comparisons will be re-run after the Northeast Monsoon.
6. **One 12-day test period.** CIs are bootstrapped over its 200 forecast times, which are
   autocorrelated (one storm spans several samples), so the intervals are likely too narrow.
7. **Few independent flood events.** The out-of-sample flood evidence is 26 reports from about
   three storms (18, 22, 27 Sep). It is indicative, not statistical. An earlier version of the
   event scores included in-sample events and has been withdrawn (lesson L029/L031, `definition_of_done.md`).
8. **CRPS against persistence can mislead.** It rewards smooth weak rain because persistence is
   penalised twice when storms move. It is always reported with a placement score (FSS) here.
9. **Resolution vs skill.** The grid is 0.29 km, but skill is only demonstrated at 6–24 km
   neighbourhoods. Do not read individual pixels as forecasts.

## Ground truth and data caveats

10. **Flood labels are threshold crossings and reports, not continuous measurements.** A PUB flood-risk alert
    means a drain sensor reached 90% of its depth — we never see levels below that. Flash-flood reports
    depend on someone observing and reporting. Absence of a report is not absence of flooding, and only
    places with sensors can raise an alert.
11. **Locations are points.** Road names are geocoded to one point (junctions to the first road
    named); PUB's flood-prone list is names, not areas.
12. **Radar is a proxy for rain at the ground.** It is NEA's colour-coded product at 33 intensity
    levels, capped at 100 mm/hr; there is no gauge calibration yet.
13. **Data cannot be redistributed.** The radar archive is built from NEA images that NEA keeps for
    about 7 days (70 km) or 30 days (240 km) and does not license for redistribution, so it is not
    published. The scripts rebuild it only for dates within NEA's retention window — **past data
    cannot be recovered from the code alone.**

## Engineering caveats

14. **Single machine, Windows.** Collection runs as Windows Scheduled Tasks on one laptop; days when
    it was off longer than NEA's retention are permanent gaps (see `scripts/check_radar_gaps.py`).
15. **Not reproducible bit-for-bit across hardware.** Sampling is seeded and reproduces exactly on the
    same GPU/driver; other hardware may differ in the last digits.
