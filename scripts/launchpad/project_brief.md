# NVIDIA LaunchPad Application Brief
# Singapore Equatorial CorrDiff Fine-tuning

---

## Project Summary

We are fine-tuning NVIDIA's CorrDiff weather downscaling model on Singapore's tropical climate — the first equatorial adaptation of this architecture.

CorrDiff was originally trained on Taiwan (23°N subtropical domain) and produces high-resolution 2km precipitation fields from coarse 25km ERA5 input. Singapore sits at 1.3°N, in a completely different convective regime: near-daily afternoon thunderstorms, sea-breeze convergence, and flash floods triggered by highly localised convective cells. Zero-shot CorrDiff inference on Singapore ERA5 data produces physically implausible outputs because it has never seen equatorial convection patterns.

The goal is to fine-tune CorrDiff on a paired ERA5 → NEA X-band radar dataset collected over Singapore from May–August 2026, producing the first probabilistic, high-resolution weather downscaling model purpose-built for Southeast Asian tropical conditions.

---

## Why This Matters

Singapore's flash floods are a recurring public safety issue. The Meteorological Service Singapore (MSS) operates a 2-hour warning service, but neighbourhood-level 30–90 minute probabilistic forecasts don't exist. A fine-tuned CorrDiff producing 2km ensemble precipitation fields over Singapore would enable:

- Sub-neighbourhood early warning for PUB (national water agency) flood response teams
- Probabilistic alerts for specific flood-prone catchments (Bukit Timah, Jurong, Bedok)
- A replicable methodology for other Southeast Asian cities (KL, Bangkok, Jakarta) facing identical convective regimes

---

## Dataset

| Component | Details |
|---|---|
| ERA5 (input) | 2022–2023, 8,760 h/year, 0.25° resolution, 8 surface + 4×4 pressure-level variables, Singapore bounding box (1.0–1.6°N, 103.5–104.1°E) |
| NEA radar (target) | NEA X-band composite, ~1km resolution, 5-min intervals. Collecting from May 2026 to August 2026 (90 days = ~25,920 frames). Colour PNG → mm/hr conversion pipeline built and validated. |
| Paired format | ERA5 6-hourly snapshots matched to nearest radar composite. Precipitation variable only (tp). |

Both datasets are already downloaded, preprocessed, and stored in zarr format. Fine-tuning begins once the 90-day radar archive is complete (~August 20, 2026).

---

## Compute Requirement

Local hardware (NVIDIA RTX 4060 Laptop, 8GB VRAM) is sufficient for the radar-only diffusion nowcaster (Stage 4, 25M params). It is **not** sufficient for CorrDiff fine-tuning: the full CorrDiff architecture is ~400M parameters and requires multi-GPU A100/H100 nodes for training on the paired ERA5+radar dataset.

LaunchPad access to NVIDIA DGX Cloud or equivalent A100 infrastructure would allow us to:
1. Run the full CorrDiff fine-tuning in ~1–2 weeks
2. Produce a Singapore-domain checkpoint compatible with earth2studio for operational inference
3. Compare fine-tuned CorrDiff against our radar-only DDPM nowcaster on the same evaluation set

---

## Technical Approach

Fine-tuning follows the CorrDiff paper's approach (Mardani et al. 2023):
- **Input:** ERA5 25km fields (same variables used in Taiwan model) cropped to Singapore domain
- **Target:** NEA radar at ~1km, precipitation (mm/hr) only
- **Architecture:** CorrDiff's score-based diffusion with regression + residual correction heads, pre-trained weights loaded from NGC
- **Fine-tuning strategy:** Freeze the regression head, fine-tune the diffusion correction network on Singapore pairs
- **Evaluation:** FSS vs persistence and vs zero-shot CorrDiff at 2/5/10 mm/hr thresholds

---

## Team & Existing Work

- Solo researcher / developer, Singapore
- NVIDIA developer account active (NGC API key)
- earth2studio 0.15.0 + physicsnemo installed, PrecipitationAFNO baseline inference already running
- Radar collection pipeline, ERA5 download, colour→mm/hr conversion, and evaluation framework all implemented
- Stage 4 DDPM nowcaster (25M params, local GPU) smoke-tested successfully June 2026

---

## Deliverables

1. Singapore-domain CorrDiff checkpoint (shareable to NVIDIA / community)
2. Evaluation report: FSS curves at 30/60/90 min lead times vs zero-shot CorrDiff and persistence
3. Open-source inference notebook for operational Singapore nowcasting
4. Technical writeup suitable for NVIDIA blog or conference submission (ICLR/NeurIPS weather workshop)

---

## Application Notes

- LaunchPad access needed: August 2026 (when 90-day radar archive is complete)
- Compute estimate: 4× A100 80GB, ~1–2 weeks of training
- No proprietary data: all sources are public (ERA5/Copernicus, NEA weather.gov.sg)
