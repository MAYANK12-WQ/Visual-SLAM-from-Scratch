"""
Camera models and utilities

Implements pinhole camera model with distortion.
"""

import numpy as np


class PinholeCamera:
    """
    Pinhole camera model.

    Intrinsic parameters:
        fx, fy: Focal lengths
        cx, cy: Principal point
        k1, k2, p1, p2, k3: Distortion coefficients (optional)

    Camera matrix K:
        [fx  0  cx]
        [0  fy  cy]
        [0   0   1]
    """

    def __init__(self, fx, fy, cx, cy, k1=0, k2=0, p1=0, p2=0, k3=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    def project(self, points_3d):
        """
        Project 3D points to image plane.

        Args:
            points_3d: Nx3 array of 3D points in camera frame

        Returns:
            Nx2 array of 2D image coordinates
        """
        # Normalize by depth
        points_2d = points_3d[:, :2] / points_3d[:, 2:3]

        # Apply intrinsics
        points_2d[:, 0] = self.fx * points_2d[:, 0] + self.cx
        points_2d[:, 1] = self.fy * points_2d[:, 1] + self.cy

        return points_2d

    def unproject(self, points_2d, depths):
        """
        Unproject 2D points to 3D given depth.

        Args:
            points_2d: Nx2 array of image coordinates
            depths: N array of depth values

        Returns:
            Nx3 array of 3D points
        """
        # Normalize coordinates
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy

        # Scale by depth
        points_3d = np.stack([x * depths, y * depths, depths], axis=1)

        return points_3d

    def pixel_to_camera(self, u, v, depth=1.0):
        """
        Convert single pixel to camera coordinates.

        Args:
            u, v: Pixel coordinates
            depth: Depth value (default: 1.0)

        Returns:
            3D point in camera frame
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])

    def camera_to_pixel(self, point_3d):
        """
        Convert 3D camera point to pixel coordinates.

        Args:
            point_3d: 3D point [x, y, z]

        Returns:
            Pixel coordinates [u, v]
        """
        x, y, z = point_3d
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return np.array([u, v])


class StereoCamera:
    """
    Stereo camera model.

    Args:
        left_camera: Left PinholeCamera
        right_camera: Right PinholeCamera
        baseline: Distance between cameras (meters)
    """

    def __init__(self, left_camera, right_camera, baseline):
        self.left = left_camera
        self.right = right_camera
        self.baseline = baseline

    def triangulate(self, pts_left, pts_right):
        """
        Triangulate 3D points from stereo correspondences.

        Args:
            pts_left: Nx2 array of points in left image
            pts_right: Nx2 array of points in right image

        Returns:
            Nx3 array of 3D points
        """
        # Compute disparities
        disparities = pts_left[:, 0] - pts_right[:, 0]

        # Avoid division by zero
        disparities = np.clip(disparities, 0.1, None)

        # Compute depth
        depths = self.left.fx * self.baseline / disparities

        # Unproject to 3D
        points_3d = self.left.unproject(pts_left, depths)

        return points_3d


if __name__ == "__main__":
    # Test camera
    camera = PinholeCamera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157)

    # Test projection
    points_3d = np.array([[1, 0, 5], [0, 1, 5], [-1, -1, 10]])
    points_2d = camera.project(points_3d)
    print(f"Projected points:\n{points_2d}")

    # Test unprojection
    depths = points_3d[:, 2]
    reconstructed = camera.unproject(points_2d, depths)
    print(f"Reconstructed points:\n{reconstructed}")
    print(f"Error: {np.linalg.norm(reconstructed - points_3d)}")
