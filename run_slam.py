"""
Main SLAM execution script

Runs Visual SLAM on various datasets or live video.
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from slam.frontend import Frontend
from utils.camera import PinholeCamera


class VisualSLAM:
    """
    Complete Visual SLAM system.

    Args:
        camera: Camera model
        config: Configuration dictionary
    """

    def __init__(self, camera, config=None):
        self.camera = camera
        self.config = config or {}

        # Initialize frontend
        self.frontend = Frontend(
            camera,
            detector_type=self.config.get('detector', 'orb'),
            num_features=self.config.get('num_features', 2000)
        )

        # Trajectory storage
        self.trajectory = []
        self.keyframe_poses = []

        # Visualization
        self.fig = None
        self.ax = None

    def process_image(self, image):
        """
        Process single image frame.

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            dict: Processing results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Process frame
        result = self.frontend.process_frame(gray)

        if result is None:
            return None

        # Store trajectory
        pose = result['pose']
        position = pose[:3, 3]
        self.trajectory.append(position)

        if result['is_keyframe']:
            self.keyframe_poses.append(pose)

        return result

    def visualize_trajectory(self):
        """Visualize estimated trajectory."""
        if len(self.trajectory) < 2:
            return

        trajectory = np.array(self.trajectory)

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Z (m)')
            self.ax.set_title('SLAM Trajectory (Top View)')
            self.ax.grid(True)
            self.ax.axis('equal')

        self.ax.clear()
        self.ax.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory')
        self.ax.plot(trajectory[0, 0], trajectory[0, 2], 'go', markersize=10, label='Start')
        self.ax.plot(trajectory[-1, 0], trajectory[-1, 2], 'ro', markersize=10, label='Current')

        # Plot keyframes
        if len(self.keyframe_poses) > 0:
            kf_positions = np.array([kf[:3, 3] for kf in self.keyframe_poses])
            self.ax.plot(kf_positions[:, 0], kf_positions[:, 2], 'k.', markersize=5, label='Keyframes')

        self.ax.legend()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_title(f'SLAM Trajectory (Top View) - {len(self.trajectory)} frames')
        self.ax.grid(True)

        plt.pause(0.001)

    def run_on_video(self, video_path, visualize=True):
        """
        Run SLAM on video file.

        Args:
            video_path (str): Path to video file
            visualize (bool): Show visualization
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        frame_count = 0
        print("Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            result = self.process_image(frame)

            if result is None:
                continue

            # Display info
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: "
                      f"Tracked={result.get('num_tracked', 0)}, "
                      f"Inliers={result.get('num_inliers', 0)}, "
                      f"Keyframe={result['is_keyframe']}")

            # Visualize
            if visualize and frame_count % 5 == 0:
                self.visualize_trajectory()

            # Show image with features
            if visualize:
                vis_frame = frame.copy()
                if result['keypoints'] is not None:
                    cv2.drawKeypoints(vis_frame, result['keypoints'], vis_frame,
                                    color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                cv2.imshow('SLAM', cv2.resize(vis_frame, (960, 540)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Keyframes: {len(self.keyframe_poses)}")
        print(f"Trajectory length: {self.get_trajectory_length():.2f} m")

        # Show final trajectory
        if visualize:
            plt.ioff()
            self.visualize_trajectory()
            plt.show()

    def run_on_images(self, image_dir, visualize=True):
        """
        Run SLAM on image sequence.

        Args:
            image_dir (str): Directory containing images
            visualize (bool): Show visualization
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

        if len(image_files) == 0:
            print(f"Error: No images found in {image_dir}")
            return

        print(f"Found {len(image_files)} images")

        for i, img_path in enumerate(image_files):
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            # Process frame
            result = self.process_image(image)

            if result is None:
                continue

            # Display info
            if i % 10 == 0:
                print(f"Image {i}/{len(image_files)}: "
                      f"Tracked={result.get('num_tracked', 0)}, "
                      f"Keyframe={result['is_keyframe']}")

            # Visualize
            if visualize and i % 5 == 0:
                self.visualize_trajectory()

        print("\nProcessing complete!")
        print(f"Keyframes: {len(self.keyframe_poses)}")
        print(f"Trajectory length: {self.get_trajectory_length():.2f} m")

        # Show final trajectory
        if visualize:
            plt.ioff()
            self.visualize_trajectory()
            plt.show()

    def get_trajectory_length(self):
        """Calculate total trajectory length."""
        if len(self.trajectory) < 2:
            return 0.0

        trajectory = np.array(self.trajectory)
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return distances.sum()

    def save_trajectory(self, output_path):
        """Save trajectory to file."""
        if len(self.keyframe_poses) == 0:
            print("No trajectory to save")
            return

        with open(output_path, 'w') as f:
            for pose in self.keyframe_poses:
                # Save as 3x4 matrix (rotation + translation)
                line = ' '.join([f'{x:.6f}' for x in pose[:3, :].flatten()])
                f.write(line + '\n')

        print(f"Trajectory saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visual SLAM')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--images', type=str, help='Image directory')
    parser.add_argument('--camera-fx', type=float, default=718.856)
    parser.add_argument('--camera-fy', type=float, default=718.856)
    parser.add_argument('--camera-cx', type=float, default=607.1928)
    parser.add_argument('--camera-cy', type=float, default=185.2157)
    parser.add_argument('--detector', type=str, default='orb', choices=['orb', 'sift'])
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--output', type=str, help='Output trajectory file')
    args = parser.parse_args()

    # Create camera
    camera = PinholeCamera(
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy
    )

    # Create SLAM system
    slam = VisualSLAM(camera, config={'detector': args.detector})

    # Run SLAM
    if args.video:
        slam.run_on_video(args.video, visualize=not args.no_viz)
    elif args.images:
        slam.run_on_images(args.images, visualize=not args.no_viz)
    else:
        print("Error: Please specify --video or --images")
        return

    # Save trajectory
    if args.output:
        slam.save_trajectory(args.output)


if __name__ == "__main__":
    main()
