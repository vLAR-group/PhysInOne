import torch
import numpy as np
import open3d as o3d


def o3d_knn(pts, num_knn):
    # https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/main/helpers.py#L72
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def mini_batch_knn(pts1, pts2, num_knn, bs=4096):
    def helper(pts1, pts2, topk):
        dist = torch.norm(pts1[:, None]-pts2[None], dim=-1, p=None)
        out = dist.topk(topk+1, largest=False)
        values, indices = out.values, out.indices
        return values[:, 1:], indices[:, 1:]
    n = pts1.shape[0]
    num_batch = round(n / bs)
    value_list, index_list = [], []
    for i in range(num_batch):
        if i < num_batch - 1:
            values, indices = helper(pts1[i*bs:(i+1)*bs], pts2, num_knn)
        else:
            values, indices = helper(pts1[i*bs:], pts2, num_knn)
        value_list.append(values)
        index_list.append(indices)
    
    if len(value_list) == 0:
        values, indices = None, None
    else:
        values, indices = torch.cat(value_list, dim=0), torch.cat(index_list, dim=0)
    return values, indices


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()
