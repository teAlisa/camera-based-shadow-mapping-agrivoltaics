# Camera-Based Shadow Mapping for Agrivoltaics

**Author:** Alisa Teige, Department of Computer Science, Michigan Technological University<br>
**Faculty advisor:** Anna Stuhlmacher (Electrical and Computer Engineering)<br>
**Technical mentor:** Gabriel Draughon (Engineering Fundamentals)

This project estimates how much sun and shade the ground under and between solar panels receives over the day. It uses time-lapse images from a single fixed camera at an agrivoltaic site: it finds the ground, detects shadows in each image, projects them onto a real-world ground grid, and summarizes shade patterns for panel rows and individual plant beds.

## Pipeline

```text
camera images ──► 01 metadata + solar position (pvlib)
              ──► 02 ground mask (SAM + fixed ROI)
              ──► 03 shadow probability maps (illumination-normalized ratio)
              ──► 04 projection onto a ground grid (camera model, cm)
              ──► 05 summary maps with panel rows
              ──► 06 sun / shade time for a selected plant bed
```

## Repository structure

```text
├── scripts/                  # run these in order
│   ├── 01_create_metadata.py
│   ├── 02_create_sam_roi_masks.py
│   ├── 03_generate_shadow_probability_maps.py
│   ├── 04_project_to_ground_grid.py
│   ├── 05_make_summary_maps.py
│   ├── 06_bed_sun_exposure.py
│   └── findCoordinates.py    # helper: click on an image to get pixel coordinates
├── src/                      # reusable modules
│   ├── metadata.py           # filenames → timestamps, solar position, daytime filter
│   ├── sam_masks.py          # SAM ground masks + ROI + QC
│   ├── shadow_model.py       # shadow probability model
│   ├── camera_geometry.py    # camera model, world ↔ image projection
│   ├── ground_grid.py        # ground grid statistics
│   ├── panel_rows.py         # solar panel row locations
│   └── visualization.py      # figures
├── sample_data/
│   ├── images/               # 14 sample images (5 July 2026, 08:51–11:01)
│   └── camera_model/         # calibrated camera model (.npz)
├── sample_outputs/           # created when you run the pipeline
├── config_example.yaml       # main parameters in one place
└── requirements.txt
```

## Quick start

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt

python scripts/01_create_metadata.py
python scripts/02_create_sam_roi_masks.py
python scripts/03_generate_shadow_probability_maps.py
python scripts/04_project_to_ground_grid.py
python scripts/05_make_summary_maps.py
python scripts/06_bed_sun_exposure.py
```

Paths are resolved relative to the repository, so the scripts can be run from any folder. Step 02 uses a SAM model through `ultralytics`; the model weights are downloaded automatically on first run.

The camera pose (`sample_data/camera_model/APS_SP_camera_model.npz`) was estimated with camera calibration code shared by Gabriel Draughon. Camera intrinsics are approximated from the horizontal field of view (70°) and a radial distortion coefficient (see `config_example.yaml`).

## What each step does

| Step | Script | Output |
|---|---|---|
| 1 | `01_create_metadata.py` — reads timestamps from filenames, adds solar elevation/azimuth, keeps daytime images | `sample_data/metadata/image_metadata_all.csv`, `image_metadata.csv` |
| 2 | `02_create_sam_roi_masks.py` — segments the visible ground with SAM, limits it to a fixed ROI, runs mask QC | `sample_data/masks/`, `sample_outputs/qc/` |
| 3 | `03_generate_shadow_probability_maps.py` — per-pixel shadow probability from an illumination-normalized image ratio, plus binary shadow masks | `sample_outputs/shadow_probability_maps/` (`.png` for viewing, `.npy` for analysis), `sample_outputs/pred_masks/` |
| 4 | `04_project_to_ground_grid.py` — projects each 50 cm ground cell into the image with the camera model and aggregates shadow statistics | `sample_outputs/grid_results/` |
| 5 | `05_make_summary_maps.py` — average shadow probability and shade-frequency maps with panel rows overlaid | `sample_outputs/figures/` |
| 6 | `06_bed_sun_exposure.py` — sunlit vs. shaded time and sunlit intervals for one plant bed | `sample_outputs/bed_sun_exposure/` |

## Adapting to a new camera view

Some coordinates are selected by hand. Run `python scripts/findCoordinates.py`, click on the image, and copy the printed pixel coordinates into:

```text
BED_POLYGON                     scripts/06_bed_sun_exposure.py
DEFAULT_SAM_POINTS              src/sam_masks.py
DEFAULT_PANEL_ROWS_PIXELS       src/panel_rows.py
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2  scripts/02_create_sam_roi_masks.py
```

If the camera moves, update these values before rerunning the pipeline.

## Acknowledgments

Thanks to Gabriel Draughon for technical guidance throughout the project and for sharing the camera calibration code and helping adapt it for this setup, and to Anna Stuhlmacher for advising this research.

## Contact

Alisa Teige — ateige@mtu.edu
