import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d

from nnutils.chamfer_distance import ChamferDistance 


def visualize_occlusion_mask(occlusion_mask, world2grid):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]

    # Generate voxel indices.
    x = torch.arange(dim_x, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    y = torch.arange(dim_y, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    z = torch.arange(dim_z, dtype=occlusion_mask.dtype, device=occlusion_mask.device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_xyz = torch.cat([
        grid_x.view(dim_x, dim_y, dim_z, 1),
        grid_y.view(dim_x, dim_y, dim_z, 1),
        grid_z.view(dim_x, dim_y, dim_z, 1)
    ], dim=3)

    # Filter visible points.
    grid_xyz = grid_xyz[occlusion_mask > 0.5]
    num_occluded_voxels = grid_xyz.shape[0]

    # Transform voxels to world space.
    grid2world = torch.inverse(world2grid)
    R_grid2world = grid2world[:3, :3].view(1, 3, 3).expand(num_occluded_voxels, -1, -1)
    t_grid2world = grid2world[:3, 3].view(1, 3, 1).expand(num_occluded_voxels, -1, -1)
    
    grid_xyz_world = (torch.matmul(R_grid2world, grid_xyz.view(-1, 3, 1)) + t_grid2world).view(-1, 3)
    
    return grid_xyz_world


def filter_occluded_points(points_pred, world2grid, occlusion_mask):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]
    num_points_pred = points_pred.shape[0]

    # Transform points to bbox space.
    R_world2grid = world2grid[:3, :3].view(1, 3, 3).expand(num_points_pred, -1, -1)
    t_world2grid = world2grid[:3, 3].view(1, 3, 1).expand(num_points_pred, -1, -1)
    
    points_pred_coords = (torch.matmul(R_world2grid, points_pred.view(num_points_pred, 3, 1)) + t_world2grid).view(num_points_pred, 3)

    # Normalize to [-1, 1]^3 space.
    # The world2grid transforms world positions to voxel centers, so we need to
    # use "align_corners=True".
    points_pred_coords[:, 0] /= (dim_x - 1)
    points_pred_coords[:, 1] /= (dim_y - 1)
    points_pred_coords[:, 2] /= (dim_z - 1)
    points_pred_coords = points_pred_coords * 2 - 1

    # Trilinearly interpolate occlusion mask.
    # Occlusion mask is given as (x, y, z) storage, but the grid_sample method
    # expects (c, z, y, x) storage.
    visibility_mask = 1 - occlusion_mask.view(dim_x, dim_y, dim_z)
    visibility_mask = visibility_mask.permute(2, 1, 0).contiguous()
    visibility_mask = visibility_mask.view(1, 1, dim_z, dim_y, dim_x)

    points_pred_coords = points_pred_coords.view(1, 1, 1, num_points_pred, 3)

    points_pred_visibility = torch.nn.functional.grid_sample(
        visibility_mask, points_pred_coords.cpu(), mode='bilinear', padding_mode='zeros', align_corners=True
    ).cuda()

    points_pred_visibility = points_pred_visibility.view(num_points_pred)

    eps = 1e-5
    points_pred_visibility = points_pred_visibility >= 1 - eps

    # Filter occluded predicted points.
    if points_pred_visibility.sum() == 0:
        # If no points are visible, we keep the original points, otherwise
        # we would penalize the sample as if nothing is predicted.
        print("All points occluded, keeping all predicted points!")
        points_pred_visible = points_pred.clone()
    else:
        points_pred_visible = points_pred[points_pred_visibility]

    return points_pred_visible


def main():
    #####################################################################################
    # Settings.
    #####################################################################################
    dist_threshold = 0.05
    max_dist = 1.0
    num_points_samples = 200000

    #####################################################################################
    # Parse command line arguments.
    #####################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_dir', action='store', dest='groundtruth_dir', default='./data/groundtruth', help='Provide root directory of ground truth data')
    parser.add_argument('--prediction_dir', action='store', dest='prediction_dir', default='./data/reconstructions', help='Provide root directory of prediction data')

    args = parser.parse_args()

    groundtruth_dir = args.groundtruth_dir
    prediction_dir = args.prediction_dir
    assert os.path.exists(groundtruth_dir)

    #####################################################################################
    # Evaluate every scene.
    #####################################################################################
    # Metrics
    acc_sum = 0.0
    compl_sum = 0.0
    chamfer_sum = 0.0
    prc_sum = 0.0
    rec_sum = 0.0
    f1_score_sum = 0.0

    total_num_scenes = 0
    scene_stats = []

    chamfer_dist = ChamferDistance()

    scene_ids = sorted(os.listdir(groundtruth_dir))
    
    for scene_id in tqdm(scene_ids):
        # Load predicted mesh.
        missing_scene = False
        mesh_pred_path = os.path.join(prediction_dir, "{}.ply".format(scene_id))
        if not os.path.exists(mesh_pred_path):
            # We have no extracted geometry, so we use default metrics for missing scene.
            missing_scene = True

        else:
            mesh_pred = o3d.io.read_triangle_mesh(mesh_pred_path)
            if np.asarray(mesh_pred.vertices).shape[0] <= 0 or np.asarray(mesh_pred.triangles).shape[0] <= 0:
                # No vertices or faces present.
                missing_scene = True

        # If no result is present for the scene, we use the maximum errors.
        if missing_scene:
            # We use default metrics for missing scene.
            print("Missing scene reconstruction: {0}".format(mesh_pred_path))
            acc_sum += max_dist
            compl_sum += max_dist
            chamfer_sum += max_dist
            prc_sum += 1.0
            rec_sum += 0.0
            f1_score_sum += 0.0
            
            total_num_scenes += 1
            continue

        # Load groundtruth mesh.
        mesh_gt_path = os.path.join(groundtruth_dir, scene_id, "mesh_gt.ply".format(scene_id))

        mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)
        points_gt = np.asarray(mesh_gt.vertices)

        # To have a fair comparison even in the case of different mesh resolutions,
        # we always sample consistent amount of points on predicted mesh.
        pcd_pred = mesh_pred.sample_points_uniformly(number_of_points=num_points_samples)
        points_pred = np.asarray(pcd_pred.points)

        # Load occlusion mask grid, with world2grid transform.
        occlusion_mask_path = os.path.join(groundtruth_dir, scene_id, "occlusion_mask.npy")
        occlusion_mask = np.load(occlusion_mask_path)

        world2grid_path = os.path.join(groundtruth_dir, scene_id, "world2grid.txt")
        world2grid = np.loadtxt(world2grid_path)

        # Put data to device memory.
        points_pred = torch.from_numpy(points_pred).float().cuda()
        points_gt = torch.from_numpy(points_gt).float().cuda()
        world2grid = torch.from_numpy(world2grid).float().cuda()

        # We keep occlusion mask on host memory, since it can be very large for big scenes.
        occlusion_mask = torch.from_numpy(occlusion_mask).float()

        # Compute gt -> predicted distance.
        dist2_gt2pred, _ = chamfer_dist(points_gt.unsqueeze(0), points_pred.unsqueeze(0))
    
        dist_gt2pred = torch.sqrt(dist2_gt2pred)
        dist_gt2pred[~torch.isfinite(dist_gt2pred)] = 0.0 # sqrt() operation is undefined for distance == 0

        dist_gt2pred = torch.minimum(dist_gt2pred, max_dist * torch.ones_like(dist_gt2pred))

        # Compute predicted -> gt distance.
        # All occluded predicted points should be masked out for , to not
        # penalize completion beyond groundtruth. 
        points_pred_visible = filter_occluded_points(points_pred, world2grid, occlusion_mask)

        if points_pred_visible.shape[0] > 0:
            dist2_pred2gt, _ = chamfer_dist(points_pred_visible.unsqueeze(0), points_gt.unsqueeze(0))
        
            dist_pred2gt = torch.sqrt(dist2_pred2gt)
            dist_pred2gt[~torch.isfinite(dist_pred2gt)] = 0.0 # sqrt() operation is undefined for distance == 0

            dist_pred2gt = torch.minimum(dist_pred2gt, max_dist * torch.ones_like(dist_pred2gt))

        # Geometry accuracy/completion/Chamfer.
        if points_pred_visible.shape[0] > 0:
            acc = torch.mean(dist_pred2gt).item() 
        else:
            acc = max_dist

        compl = torch.mean(dist_gt2pred).item()
        chamfer = 0.5 * (acc + compl)

        # Precision/recall/F1 score.
        if points_pred_visible.shape[0] > 0:
            prc = (dist_pred2gt <= dist_threshold).float().mean().item()
        else:
            prc = 0.0
            
        rec = (dist_gt2pred <= dist_threshold).float().mean().item()

        if prc + rec > 0:
            f1_score = 2 * prc * rec / (prc + rec)
        else:
            f1_score = 0.0

        # print("acc =", acc)
        # print("compl =", compl)
        # print("chamfer =", chamfer)
        # print("prc =", prc)
        # print("rec =", rec)
        # print("f1_score =", f1_score)

        # Update total metrics.
        acc_sum += acc
        compl_sum += compl
        chamfer_sum += chamfer
        prc_sum += prc
        rec_sum += rec
        f1_score_sum += f1_score

        total_num_scenes += 1

        # Update scene stats.
        scene_stats.append({
            "scene_id": scene_id,
            "acc": acc,
            "compl": compl,
            "chamfer": chamfer,
            "prc": prc,
            "rec": rec,
            "f1_score": f1_score
        })

        # Just for debugging: Visualize occluded points.
        # occluded_pcd = o3d.geometry.PointCloud()
        # occluded_pcd.points = o3d.utility.Vector3dVector(visualize_occlusion_mask(occlusion_mask, world2grid).cpu().numpy())
        # occluded_pcd.paint_uniform_color([0.7, 0.0, 0.0])

        # o3d.visualization.draw_geometries([mesh_gt, occluded_pcd], mesh_show_back_face=True)

    #####################################################################################
    # Report evaluation results.
    #####################################################################################
    # Report independent scene stats.
    # Sort by speficied metric.
    sorted_idxs = [i[0] for i in sorted(enumerate(scene_stats), key=lambda x:-x[1]["f1_score"])]

    print()
    print("#" * 50)
    print("SCENE STATS")
    print("#" * 50)
    print()

    num_best_scenes = 20

    for i, idx in enumerate(sorted_idxs):
        if i >= num_best_scenes:
            break

        print("Scene {0}: acc = {1}, compl = {2}, chamfer = {3}, prc = {4}, rec = {5}, f1_score = {6}".format(
            scene_stats[idx]["scene_id"], 
            scene_stats[idx]["acc"], scene_stats[idx]["compl"], scene_stats[idx]["chamfer"], 
            scene_stats[idx]["prc"], scene_stats[idx]["rec"], scene_stats[idx]["f1_score"]
        ))

    # Metrics summary.
    mean_acc = acc_sum / total_num_scenes
    mean_compl = compl_sum / total_num_scenes
    mean_chamfer = chamfer_sum / total_num_scenes
    mean_prc = prc_sum / total_num_scenes
    mean_rec = rec_sum / total_num_scenes
    mean_f1_score = f1_score_sum / total_num_scenes

    metrics = {
        "acc": mean_acc,
        "compl": mean_compl,
        "chamfer": mean_chamfer,
        "prc": mean_prc,
        "rec": mean_rec,
        "f1_score": mean_f1_score
    }

    print()
    print("#" * 50)
    print("EVALUATION SUMMARY")
    print("#" * 50)
    print("{:<30} {}".format("GEOMETRY ACCURACY:",      metrics["acc"]))
    print("{:<30} {}".format("GEOMETRY COMPLETION:",    metrics["compl"]))
    print("{:<30} {}".format("CHAMFER:",                metrics["chamfer"]))
    print("{:<30} {}".format("PRECISION:",              metrics["prc"]))
    print("{:<30} {}".format("RECALL:",                 metrics["rec"]))
    print("{:<30} {}".format("F1_SCORE:",               metrics["f1_score"]))

    
if __name__=="__main__":
    main()