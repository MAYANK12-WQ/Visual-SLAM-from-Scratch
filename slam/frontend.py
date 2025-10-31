"""
SLAM Frontend - Visual Odometry and Tracking

Handles:
- Feature detection and matching
- Camera pose estimation
- Map point tracking
- Keyframe selection
"""

import cv2
import numpy as np
from collections import deque


class Frontend:
    """
    SLAM Frontend for visual odometry.

    Args:
        camera: Camera model
        detector_type (str): Feature detector type ('orb', 'sift')
        num_features (int): Maximum number of features
    """

    def __init__(self, camera, detector_type='orb', num_features=2000):
        self.camera = camera
        self.num_features = num_features

        # Feature detector
        if detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=num_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif detector_type == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=num_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        # Tracking state
        self.current_frame = None
        self.last_frame = None
        self.keyframes = []

        # Statistics
        self.num_tracked = 0
        self.num_inliers = 0

    def process_frame(self, image):
        """
        Process new image frame.

        Args:
            image: Input image (grayscale)

        Returns:
            dict: {
                'pose': 4x4 transformation matrix,
                'keypoints': detected keypoints,
                'descriptors': feature descriptors,
                'is_keyframe': bool
            }
        """
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        # Initialize first frame
        if self.last_frame is None:
            self.last_frame = {
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pose': np.eye(4)
            }
            return {
                'pose': np.eye(4),
                'keypoints': keypoints,
                'descriptors': descriptors,
                'is_keyframe': True
            }

        # Match features with last frame
        matches = self.match_features(self.last_frame['descriptors'], descriptors)

        if len(matches) < 10:
            print("Warning: Too few matches")
            return None

        # Extract matched points
        pts1 = np.float32([self.last_frame['keypoints'][m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Estimate relative pose
        relative_pose, inliers = self.estimate_pose(pts1, pts2)

        if relative_pose is None:
            print("Warning: Pose estimation failed")
            return None

        # Update absolute pose
        current_pose = self.last_frame['pose'] @ relative_pose

        # Check if keyframe needed
        is_keyframe = self.is_keyframe_needed(inliers, relative_pose)

        # Store current frame
        self.current_frame = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': current_pose
        }

        # Update tracking
        if is_keyframe:
            self.keyframes.append(self.current_frame)
            self.last_frame = self.current_frame
        else:
            self.last_frame['pose'] = current_pose

        self.num_tracked = len(matches)
        self.num_inliers = int(inliers.sum())

        return {
            'pose': current_pose,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'is_keyframe': is_keyframe,
            'num_tracked': self.num_tracked,
            'num_inliers': self.num_inliers
        }

    def match_features(self, desc1, desc2):
        """
        Match features between frames.

        Args:
            desc1: Descriptors from frame 1
            desc2: Descriptors from frame 2

        Returns:
            list: Filtered matches
        """
        # KNN matching
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_pose(self, pts1, pts2):
        """
        Estimate camera pose from matched points.

        Args:
            pts1: Points in frame 1
            pts2: Points in frame 2

        Returns:
            tuple: (pose_matrix 4x4, inliers_mask)
        """
        # Essential matrix estimation
        E, inliers = cv2.findEssentialMat(
            pts1, pts2,
            self.camera.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None

        # Recover pose
        _, R, t, inliers = cv2.recoverPose(E, pts1, pts2, self.camera.K, mask=inliers)

        # Construct 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.squeeze()

        return pose, inliers

    def is_keyframe_needed(self, inliers, relative_pose):
        """
        Determine if current frame should be a keyframe.

        Criteria:
        - Low number of inliers
        - Large motion
        - Sufficient time elapsed
        """
        if len(self.keyframes) == 0:
            return True

        # Check inlier ratio
        if inliers is not None:
            inlier_ratio = inliers.sum() / len(inliers)
            if inlier_ratio < 0.7:
                return True

        # Check translation magnitude
        translation = np.linalg.norm(relative_pose[:3, 3])
        if translation > 0.1:  # >10cm movement
            return True

        # Check rotation magnitude
        R = relative_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle > np.deg2rad(10):  # >10 degree rotation
            return True

        return False


if __name__ == "__main__":
    # Test frontend
    from utils.camera import PinholeCamera

    camera = PinholeCamera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157)
    frontend = Frontend(camera, detector_type='orb')

    # Create dummy images
    img1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    result1 = frontend.process_frame(img1)
    result2 = frontend.process_frame(img2)

    print(f"Frame 1: {result1['is_keyframe']}")
    if result2:
        print(f"Frame 2: Tracked={result2['num_tracked']}, Inliers={result2['num_inliers']}")
