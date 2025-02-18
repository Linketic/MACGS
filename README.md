# MACGS: Motion-Aware Compression for Dynamic Gaussian Splatting

## Environmental Setups
In our environment, we use pytorch=2.4.1+cu118.
```bash
git submodule update --init --recursive
conda create -n macgs python=3.10
conda activate macgs
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

## Data Preparation
To save the memory, you should extract the frames of each video and then organize your dataset as follows. The dataset provide in [Neural_3D_Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) is used.
```
├── data
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```
Then, generate point clouds from input data.
```bash
bash colmap.sh data/dynerf/coffee_martini llff
```
Finally, downsample the point clouds.
```bash
python scripts/downsample_point.py data/dynerf/coffee_martini/colmap/dense/workspace/fused.ply data/dynerf/coffee_martini/points3D_downsample2.ply
```

## Training
For training scenes such as `coffee_martini`, run
```bash
python train_stage1.py -s data/dynerf/coffee_martini/ --configs arguments/dynerf/default.py --expname dynerf/coffee_martini --store_grads --mercy_points --prune_dead_points --mercy_type dynamic_static_mercy --split_dynamic
python train_stage2.py -s data/dynerf/coffee_martini --configs arguments/dynerf/default.py --expname dynerf/coffee_martini
```

## Rendering
Run the following script to render the images.
```bash
python render.py -m output/dynerf/coffee_martini --configs arguments/dynerf/default.py --skip_train --dynamic
```

## Evaluation
You can just run the following script to evaluate the model.

```bash
python metrics.py --model_path output/dynerf/coffee_martini
```
