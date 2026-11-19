# Multi-Camera Tracking: intersection dataset labels and parameter study

This repository accompanies our paper on multi-camera vehicle tracking at an urban intersection (YOLOv4 detection, DeepSORT single-camera tracking, vehicle re-identification, and cross-camera association using global location, appearance and vehicle class). It provides:

- a description of the four-camera dataset and its class distribution;
- the manual labels of a 2-minute validation clip;
- the code and results of the parameter study: a grid search and sensitivity analysis of the association weights and threshold.

The videos themselves are not included (see [Data availability](#data-availability)).

## Dataset

**Acquisition.** Four fixed cameras installed around an urban intersection in a mid-sized U.S. city recorded for one hour on 31 October 2023, between 12:00 and 13:00, in daylight. Each camera recorded at 1920 × 1080 pixels and 15 FPS (54,006 frames per camera, 216,024 in total). All views were corrected for nonlinear lens distortion, and the four videos are synchronized to within one frame over the whole hour. For two cameras (194 and 193), only the part of the image covering the intersection was processed (1114 × 626 and 1286 × 724 pixels); their boxes are given in the coordinates of the processed image.

**Scenario.** The cameras view the intersection from different angles. Parts of their views overlap, so a vehicle can be visible in several cameras at the same time; other parts do not, so a vehicle can leave one view before it enters the next.

**Vehicles and classes.** 10,042 distinct vehicles were counted manually in the hour; a vehicle seen by several cameras was counted once. Class distribution (`data/class_distribution.csv`):

| Class | Share |
|---|---|
| Passenger car | 56.5% |
| Pickup truck | 29.9% |
| Oversize truck | 6.2% |
| Van | 5.1% |
| Bus | 2.3% |

**Annotation protocol.** All annotations were made manually by one of the authors:
- the vehicle count and the class of each vehicle (each vehicle counted once, even if it appeared in several cameras);
- the identity matches (IDTP, IDFP and IDFN) used in the paper to evaluate multi-camera tracking, counted from the output videos of each method;
- the labels of the 2-minute validation clip described below.

## Data availability

The videos were collected within a U.S. Department of Energy (DOE) project and are not publicly available. They are available from the corresponding author on reasonable request, subject to approval by the project. This repository contains no video frames.

## Validation clip labels (`data/validation_clip/`)

The clip is the first 2 minutes of the four cameras (tracks frames 1 to 1800; the video frame index is frame − 1). The tracker's single-camera tracks were cut into segments wherever a track had no row for 8 frames or more. Each segment has a key `uid_firstframe`: for example, `25_16` is track 25 from frame 16. The window contains 286 segments. The annotator did not see the association results and answered two kinds of question:

- 241 **match questions**: for a segment Q, which of the segments of a neighbouring camera that the tracker compared with Q, if any, shows the same vehicle;
- 56 **gap checks**: whether the vehicle before and after a gap in one track is the same.

Answers that contradicted each other were reviewed a second time (23 questions, marked in the `second_look` column).

| File | Content |
|---|---|
| `questions.json` | The 297 questions. `meta.segments_in_window` lists all segments. Each question gives the frame shown, the query and candidate segments, and their boxes per frame as `[x1, y1, x2, y2, tsu]`. |
| `answers.csv` | `n`, `qid`, `type` (`match` or `gap`), `answer` (candidate number `1` to `6` or `none` for match questions; `same` or `different` for gap checks), `flagged`, `second_look`. |
| `tracks.csv` | Tracker output for frames 1 to 1800: `frame`, `cam`, `uid` (single-camera track), `label` (global ID in the recorded run), `cls`, box `x1, y1, x2, y2` (pixels), `Xw, Yw` (global coordinates, metres), `tsu` (frames since the last detection; 0 = detected in this frame). |
| `steps.pkl` | The 4,371 recorded cross-camera matching steps (track positions, classes, appearance costs and the live decisions) used to replay the association. It is a Python pickle (a list of dicts of NumPy arrays); only load pickles from sources you trust. |

## Parameter study (`code/`, `results/`)

The association cost is put on a normalized scale:

cost = λl · (distance / 5 m) + λf · appearance + λc · class (0 for the same class, 1 otherwise), with λl + λf + λc = 1,

and a pair of tracks is matched when cost ≤ Tdis. This only rescales the cost of the live code (weights 0.4 / 0.3 / 0.3, class mismatch 0.5, threshold 2), which becomes λ = (0.816, 0.122, 0.061) and Tdis = 0.816 and produces exactly the same IDs. The separate threshold for oversize trucks (4.08 on this scale) is kept fixed. The grid covers all 231 weight combinations in steps of 0.05 and 30 thresholds from 0.05 to 1.50, for 6,930 settings. Each setting replays the recorded matching steps with the single-camera tracks fixed, and is scored by IDF1 against the labels.

To run it you need Python 3.7 or later. It was tested with numpy 1.19, pandas 1.3, scipy 1.4 and matplotlib 3.3.

```bash
pip install -r requirements.txt
python code/truth.py          # truth from the labels; lists contradictions
python code/grid.py --check   # the replay reproduces the recorded IDs; the rescale changes nothing
python code/grid.py           # all settings (about 5 minutes on 14 cores)
python code/summarize.py      # key numbers -> results/summary.json
python code/figure.py         # results/fig_sensitivity.png and .pdf
```

`code/compare.py` gives a bootstrap interval for the IDF1 difference between two settings.

**Main results** (`results/`):
- The best IDF1, 92.13%, is reached by 10 settings (λl from 0.55 to 0.95, Tdis from 0.70 to 0.85). The setting used in the paper reaches 92.07%.
- At Tdis = 0.82, IDF1 stays within 1 point of the best for 36% of the weight combinations, including every combination with λl ≥ 0.55.
- The threshold has the largest effect.

## License

The code (`code/`) is under the MIT License; see `LICENSE`. The data (`data/`) and results (`results/`) are under CC BY 4.0; see `data/LICENSE.md`.

## Citation

The citation will be added when the paper is published.
