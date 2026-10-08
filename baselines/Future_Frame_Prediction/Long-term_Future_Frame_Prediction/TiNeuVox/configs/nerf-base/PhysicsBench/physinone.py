_base_ = "./default.py"

# Set these two fields for each scene before training.
expname = "SinglePhysics/example_scene_trajectory"
basedir = "./output"

data = dict(
    datadir="/path/to/FuturePrediction/SinglePhysics/example_scene_trajectory",
    dataset_type="physicsbench",
    white_bkgd=True,
    half_res=True,
    testskip=1,
    load2gpu_on_the_fly=True,
)

train_config = dict(
    N_iters=20000,
    N_rand=4096,
)

model_and_render = dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    stepsize=0.5,
)
