# White-plush fusion reproduction

This folder contains the selected plush mesh, fixed placement transform, camera path, motion manifest, and final depth-aware orbit video.

The delivered `orbit.mp4` is a 240-frame, 10-second, 24-fps H.264 composite. The plush uses one fixed Garden-world transform while the camera moves through a one-way 180-degree half-turn. The object begins visible, passes behind the vase, and reappears. The Garden depth audit records per-frame visibility and occlusion measurements in `evaluation/fusion_depth_audit.json`.

## Reproduction requirements

Re-rendering requires the Garden step-30000 checkpoint, the selected plush mesh, the course Garden camera metadata, and the generated Garden RGB and depth frames. These large inputs and checkpoints are not part of this repository. See the top-level README for the recorded software environments. The commands below show the pipeline order; supply local paths for your data and outputs.

1. Build a fixed placement with `code/fusion/build_fixed_placement.py`.
2. Generate the camera path with `code/fusion/build_rotation_path.py`.
3. Render Garden RGB and depth with `code/fusion/render_garden_depth.py`.
4. Composite the plush mesh with `code/fusion/composite_fixed_mesh.py` (or `composite_plushie_mesh.py` for the selected vertex-color-preserving variant).
5. Encode the resulting frames with FFmpeg at 24 fps.

The object transform remains fixed across the orbit. The composite uses Garden depth for occlusion and suppresses table-depth conflicts near the support contact band. It does not synthesize cast/contact shadows or relight either reconstruction.
