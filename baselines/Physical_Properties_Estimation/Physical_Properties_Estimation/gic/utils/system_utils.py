#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import trimesh as tm
from errno import EEXIST
from os import makedirs, path
import matplotlib.pyplot as plt




def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def check_gs_model(model_path, saving_iterations, fix_pcd=False):
    file_name = "point_cloud_fix_pcd" if fix_pcd else "point_cloud"
    p = os.path.join(model_path, file_name)
    if not os.path.exists(model_path):
        return False
    if not os.path.exists(p):
        return False
    if len(os.listdir(p)) <= 0:
        return False
    if not searchForMaxIteration(p) == saving_iterations[-1]:
        return False
    else:
        max_iter = searchForMaxIteration(p)
        print(f'Find the model: {p}, iterations: {max_iter}')
        return True

def draw_curve(curve, path, name='loss'):
    x = range(1, len(curve) + 1)
    plt.figure()
    plt.plot(x, curve, label=name)

    plt.xlabel('Training Steps')
    plt.ylabel(name)
    plt.title('{} curve'.format(name))

    plt.legend()

    plt.savefig('{}/{}.png'.format(path, name))

def write_particles(particles, idx, path, name='', vertex_colors=None):
    
    if type(particles) == np.ndarray:
        numpy_array = particles
    else:
        numpy_array = particles.cpu().detach().numpy()
    if not os.path.exists(os.path.join(path, 'mpm')):
        mkdir_p(os.path.join(path, 'mpm'))
    if vertex_colors is None:
        tm.Trimesh(numpy_array).export(os.path.join(path, f'mpm/{name}_{idx}.ply'))
    else:
        tm.Trimesh(numpy_array, 
                vertex_colors=vertex_colors).export(os.path.join(path, f'mpm/{name}_{idx}.ply'))