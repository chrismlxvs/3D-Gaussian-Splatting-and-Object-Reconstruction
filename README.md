# CIE6004 Assignment 2: Garden Reconstruction and White-Plush Integration

This repository contains the project implementation, final report, rendered outputs, and validation evidence for reconstructing the Garden scene and a white plush object, then integrating the object into a camera orbit through the scene.

## Contents

- `code/garden/`: 3D Gaussian Splatting training and rendering.
- `code/object_recon/`: object pose, mask, mesh processing, and evaluation utilities. `vendor/milo/` includes the MILo source and its upstream license.
- `code/fusion/` and `fusion/`: placement, camera path, depth-aware compositing, and delivered mesh/video.
- `renders/`, `scene/`, and `fusion/orbit.mp4`: generated visual results.
- `evaluation/` and `evidence/`: validation measurements and selected review images.
- `report.pdf` and `report_src/`: report and editable LaTeX source.

The course data, pretrained model checkpoints, training environments, and assignment package are not included. Obtain permitted inputs from the course source and keep them outside Git. The final Garden renders and fusion artifacts are included as project outputs.

## Results

On the 12 public Garden validation views, the reconstruction scores 27.029 PSNR, 0.8592 SSIM, and 0.08371 LPIPS, improving on the supplied baseline. The white-plush mesh scores 0.9362 mean silhouette IoU on 12 held-out views after culling guided only by training silhouettes. The final static placement is composited against Garden depth along a one-way 180-degree camera orbit; the plush is occluded by the vase and then revealed again. These are validation and visual integration results, not test-set accuracy claims.

## Environment

The Garden pipeline was run with Python 3.10, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, gsplat 1.5.3, SciPy 1.15.3, and LPIPS 0.1.4. It requires a compatible CUDA GPU and the assignment's Garden data and COLMAP reconstruction. Install a PyTorch build matching your CUDA setup from [the official PyTorch selector](https://pytorch.org/get-started/locally/), then install the other packages and the pinned gsplat release in the project environment.

The object pipeline used a separate Python 3.9 environment with PyTorch 2.3.1+cu118, Open3D 0.19.0, SAM 2.1, and rembg 2.0.68/U2Net. Training also depends on MILo's Rade-GS rasterizer submodule and the prepared object images, masks, and camera poses. The vendored MILo license is in `code/object_recon/vendor/milo/LICENSE.md`; its source has a separate license at `code/object_recon/vendor/milo/milo/LICENSE`.

## Reproduction entry points

Garden training script:

```bash
python code/garden/train_garden.py --help
```

Garden rendering script:

```bash
python code/garden/render_garden.py --help
```

Fusion placement, path generation, depth rendering, and compositing instructions are in [`fusion/README.md`](fusion/README.md). Commands in the original research log depended on private machine paths and locally staged checkpoints; they are intentionally not copied here. Full end-to-end retraining therefore requires adapting the commands to your dataset and hardware paths.

## Report source

From `report_src/`, build the report with LaTeX:

```bash
pdflatex main.tex
pdflatex main.tex
```

The figures used by the report are in `evidence/`. Build products are ignored by Git.

## Licensing and data

No project-wide license is provided because the coursework materials and third-party components have separate terms. Review the included MILo and Gaussian Splatting licenses before redistribution. The course dataset and assignment instructions are not included or relicensed here. Add a project license only after deciding what reuse rights you want to grant.
