# Data

`metrics_acdc_val.json` contains per-patient segmentation-quality and clinical metrics for the
**ACDC validation cohort (n = 20; 4 patients per class — NOR, DCM, HCM, MINF, RV)**, for every
segmentation method evaluated in the upstream study (MedSAM2 ED/ES/Dual, SAM2, U-Net, DINOv2,
plus box-prompt-noise ablations).

This repository uses the `MedSAM2_Dual` entries. Per patient:

| Field | Meaning |
|---|---|
| `pid`, `group` | ACDC patient ID and ground-truth pathology class |
| `gt_EF`, `gt_EDV`, `gt_ESV` | LV metrics computed from the **reference** ACDC masks |
| `pred_EF`, `pred_EDV`, `pred_ESV` | LV metrics computed from **MedSAM2 dual-anchored** zero-shot masks |
| `dice_*`, `hd95_*`, `assd_*` | segmentation quality per structure (RV / Myo / LV) |

No image data is included. The underlying cine MRI comes from the public
[ACDC challenge dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) (Bernard et al.,
IEEE TMI 2018); the masks and metrics were produced by the zero-shot segmentation pipeline in
[4DCMR-Segmentation-via-Foundation-Model](https://github.com/zhuoanli/4DCMR-Segmentation-via-Foundation-Model).
