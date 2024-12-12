"""
visualize the np saved data_packets from process_slam
"""
import numpy as np
from nerfslam_utils.flow_viz import viz_depth_map, viz_depth_sigma
import cv2
import torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process depth_cov data.")
    parser.add_argument(
        '--fn',
        type=str,
        help='Path to depth_cov.npy file',
        default='/media/pi/MyBookDuo/jhuai/results/nerfslam_sample/nerfslam_data/depth_cov_96.npy'
    )
    args = parser.parse_args()
    d = np.load(args.fn, allow_pickle=True).item()

    k = d['k']
    depth = d['depths']
    depth_scale = d['depth_scale']
    depth_cov = d['depths_cov']
    depth_cov_scale = d['depths_cov_scale']

    # data_packets = {"k": viz_idx,
    #                 "poses": world_T_cam0,  # needs to be c2w
    #                 "images": images.contiguous().cpu().numpy(),
    #                 "depths": depths.contiguous().cpu().numpy(),
    #                 "depth_scale": scale,
    #                 # This should be scale, since we scale the poses... # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale,
    #                 "depths_cov": depths_cov.contiguous().cpu().numpy(),  # do not use up
    #                 "depths_cov_scale": scale,  # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale,
    #                 "gt_depths": gt_depths.contiguous().cpu().numpy(),
    #                 "calibs": calibs,
    #                 }

    print(f'depth scale {depth_scale}')
    print(f'depth cov scale {depth_cov_scale}')

    nimgs = depth.shape[0]
    for i in range(nimgs):
        dep = depth[i, :, :, 0]
        dcov = depth_cov[i, :, :, 0]
        idx = k[i]
        print(f'{idx} depth shape {dep.shape} min {np.min(dep)} max {np.max(dep)} median {np.median(dep)} mean {np.mean(dep)}')
        print(f'{idx} depth cov shape {dcov.shape} min {np.min(dcov)} max {np.max(dcov)} median {np.median(dcov)} mean {np.mean(dcov)}')
        viz_depth_sigma(torch.from_numpy(np.sqrt(dcov)).unsqueeze(0), fix_range=True, bg_img=None, sigma_thresh=20.0,
                        name="Depth Sigma for Fusion")
        cv2.waitKey(2000)
