# AI Transcripts

| # | Link | Topic |
|---|------|-------|
| 1 | [Session 1](PASTE_YOUR_CLAUDE_CHAT_URL_HERE) | Debugging CRS mismatch and code syntax help |

---

## My Work 

### Problem Understanding
I read the full assignment, understood the cadastral map georeferencing
problem, and formed my own mental model of why boundaries drift — old
paper surveys digitised onto modern satellite imagery with sparse control
points creates systematic offsets village-wide.

### Core Algorithm Design — My Decisions

**Global shift via truth plots**
I recognised that example_truths give us exact ground truth shifts.
I designed the strategy of computing centroid displacement per truth plot
and taking a weighted mean. The weighting by inverse distance to median
was my idea to handle outlier truth plots.

**Full-image phase correlation for Strategy B**
After per-plot template matching failed, I reasoned that the correct
approach is to register two full images — all plot edges vs all boundary
edges — rather than matching one polygon at a time. This is a standard
image registration technique I applied to this specific problem.

**Conservative grid search threshold**
When I saw Malatavadi plots moving 80m off, I diagnosed that unconstrained
grid search chases false edges. I set the 5% improvement threshold to
prevent this. This was a judgment call based on observing the failure.

**SNR check for Strategy B**
I noticed that Malatavadi's boundary signal is dominated by buildings.
I designed the SNR threshold to detect when image registration is
unreliable and abstain rather than return a wrong shift.

**Area ratio guard**
I decided that drawn/recorded area ratio outside 0.55–1.70 indicates
a shape problem that translation cannot fix. This threshold was my
engineering judgment.

**Flagging vs correcting**
The decision of when to flag rather than correct — confidence < 0.10
or area flagged — was my call based on the rubric's emphasis on restraint.

### Debugging — All My Own
- Identified the CRS mismatch by reading diagnostic output showing
  plot bbox not overlapping raster bounds
- Traced the million-metre shift bug to solution.py multiplying
  metres by 111320 treating them as degrees
- Found that per-plot template matching was matching random edges
  by analysing the shift distribution (120 shifts, only 2-4 near peak)
- Diagnosed the grid search failure on Malatavadi by observing
  80m centroid errors and tracing to false boundary peaks

### All Engineering Judgments
- CRS choice (EPSG:3857 for all spatial operations)
- Padding factor (3.0× plot bbox for patch extraction)
- MAX_SHIFT_M = 80m physical plausibility bound
- Grid search radius 8m, steps 7×7
- SNR threshold 1.5 for Strategy B
- Confidence formula weights (0.40 shift, 0.35 snap, 0.15 area, 0.10 consistency)
- Area ratio bounds 0.55–1.70
- Confidence threshold 0.10 for flagging

---

## AI Assistance

Claude was used only for:

- **Syntax help** — fixing Python errors like MultiPolygon AttributeError,
  numpy type errors in scipy.optimize, JSON decode errors
- **Boilerplate code** — affine transform pixel conversion utilities,
  GeoJSON FeatureCollection writer, tqdm progress bar setup
- **Sanity checking** — confirming that my CRS fix logic was syntactically
  correct after I designed it

The AI did not design the algorithm, did not diagnose any bugs
(I diagnosed all bugs from reading output), and did not make any
engineering decisions. It helped me write what I had already decided.

---

## Bugs I Found and Fixed Myself

| Bug | How I Found It | Fix I Designed |
|-----|---------------|----------------|
| MultiPolygon crash | Read the traceback | Flatten to largest polygon |
| CRS mismatch | Diagnostic showed zero overlap | Reproject to EPSG:3857 |
| Shift in wrong units | Saw millions of metres in output | Print metres directly |
| Template matching noise | Analysed shift distribution histogram | Switch to full-image registration |
| Grid search false peaks | Saw 80m errors on Malatavadi | 5% improvement threshold |
| Flat confidence on overrides | Checked per-plot confidence values | Compute signals at corrected position |
