# Garden dataset placement

Obtain the Garden training, validation, test-camera, COLMAP, and trajectory assets from the course source. Do not commit course data or test reference images to this repository. The Garden training script accepts the data root, camera JSON, and output directory explicitly:

```bash
python code/garden/train_garden.py --data-root /path/to/garden --camera-json /path/to/garden/train/cameras.json --output /path/to/output --physical-gpu 0
```

Select a compatible PyTorch/CUDA environment and bind the process to an available GPU before training. The script expects the training images, `train/cameras.json`, and supplied COLMAP `sparse/0/points3D.bin` in the layout used by the assignment release.
