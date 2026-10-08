import numpy as np
import open3d as o3d


COLOR20 = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])

COLORGRAY1 = np.array([63, 63, 63])
COLORGRAY2 = np.array([127, 127, 127])
COLORGRAY3 = np.array([192, 192, 192])
COLORGRAY4 = np.array([223, 223, 223])


def build_colored_pointcloud(pc, color):
    """
    :param pc: (N, 3).
    :param color: (N, 3).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud

def build_pointcloud_segm(pc, segm):
    """
    :param pc: (N, 3).
    :param segm: (N,).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(COLOR20[segm % COLOR20.shape[0]] / 255.)
    return point_cloud


lines = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
box_colors = [[0, 1, 0] for _ in range(len(lines))]

def build_bbox3d(boxes):
    """
    :param boxes: List [(8, 3), ...].
    """
    line_sets = []
    for corner_box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(box_colors)
        line_sets.append(line_set)
    return line_sets

def bound_to_box(bounds):
    """
    :param bounds: List [(3, 2), ...].
    """
    boxes = []
    for bound in bounds:
        corner_box = np.array([[bound[0, 0], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 1]]])
        boxes.append(corner_box)
    return boxes

def build_segm_vis(segm, with_background=False):
    """
    :param segm: (H, W).
    """
    if with_background:
        colors = np.concatenate((COLOR20[-1:], COLOR20[:-1]), axis=0)
    else:
        colors = COLOR20
    colors = colors / 255.
    segm_vis = colors[segm % COLOR20.shape[0]]
    return segm_vis


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

def pc_flow_to_sphere(pc, flow, radius=0.001, resolution=10, color=COLORGRAY2):
    """
    Visualize scene flow vectors as arrows.
    :param pc: (N, 3)
    :param flow: (N, 3)
    """
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    n_point = pc.shape[0]

    meshes = []
    for pid in range(n_point):
        point, point_flow = pc[pid], flow[pid]
        flow_len = np.linalg.norm(point_flow)

        # Fill too-small flow
        if flow_len < 1e-6:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 2, resolution=resolution)
            if len(color.shape) == 2:
                mesh.paint_uniform_color(color[pid] / 255.)
            else:
                mesh.paint_uniform_color(color / 255.)
            mesh.translate(pc[pid])

        else:
            point_flow = point_flow / flow_len
            # flow_len = np.linalg.norm(point_flow) * 10
            mesh = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=0.2 * flow_len,
                # cone_radius=1.5 * radius,
                cone_radius=2.5 * radius,
                cylinder_height=0.8 * flow_len,
                cylinder_radius=radius,
                resolution=resolution
            )
            if len(color.shape) == 2:
                mesh.paint_uniform_color(color[pid] / 255.)
            else:
                mesh.paint_uniform_color(color / 255.)
            rot_mat = caculate_align_mat(point_flow)
            mesh.rotate(rot_mat, center=(0, 0, 0))
            mesh.translate(point)

        meshes.append(mesh)

    # Merge
    mesh = meshes[0]
    for i in range(1, len(meshes)):
        mesh += meshes[i]
    return mesh

def pc_segm_to_sphere(pc, segm=None, radius = 0.01, resolution=10, with_background=False, default_color=COLORGRAY2):
    """
    Visualize point cloud as mesh balls. The color denotes hard segmentation.
    :param pc: (N, 3)
    :param segm: (N,)
    """
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    n_point = pc.shape[0]
    if with_background:
        colors = np.concatenate((COLOR20[-1:], COLOR20[:-1]), axis=0)
    else:
        colors = COLOR20

    meshes = []
    for pid in range(n_point):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        if segm is not None:
            mesh.paint_uniform_color(colors[segm[pid] % colors.shape[0]] / 255.)
        else:
            mesh.paint_uniform_color(default_color / 255.)
        mesh.translate(pc[pid])
        meshes.append(mesh)

    # Merge
    mesh = meshes[0]
    for i in range(1, len(meshes)):
        mesh += meshes[i]
    return mesh