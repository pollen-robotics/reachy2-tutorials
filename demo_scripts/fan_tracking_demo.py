import argparse
import logging
import threading
import time

import cv2  # type: ignore
import numpy as np
from huggingface_hub import hf_hub_download  # type: ignore
from reachy2_sdk import ReachySDK  # type: ignore
from supervision import Detections  # type: ignore
from ultralytics import YOLO  # type: ignore


class FaceTracker:
    def __init__(
        self,
        reachy: ReachySDK,
        threshold: float = 0.8,
        freq: float = 20.0,
        visualize: bool = False,
    ):
        """Initialize the FaceTracker, allowing the Reachy robot to track faces using a YOLO model,
        to convert the position from pixels to Reachy's frame, and to make the robot look at the face.

        Args:
            reachy (ReachySDK): The Reachy robot instance.
            threshold (float): Confidence threshold for face detection.
            freq (float): Frequency of face tracking updates in Hz.
            visualize (bool): Whether to visualize the face detection results.
        """
        self._reachy = reachy
        self._threshold = threshold
        self._freq = freq
        self._visualize = visualize

        # Init variables for face position and area
        self._position = None
        self._area = None

        self._enabled = threading.Event()

        # Get camera parameters
        cam_params = self._reachy.cameras.teleop.get_parameters()
        self._P = np.array(cam_params[6]).reshape((3, 4))
        self._height = cam_params[0]
        self._width = cam_params[1]

        # Load the YOLO model for face detection from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt"
        )
        self.model = YOLO(model_path).to("cpu")

    def start(self) -> None:
        """Start the face tracking thread."""
        self._enabled.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the face tracking thread."""
        self._enabled.clear()
        self._thread.join()

    def get_face_info(self) -> tuple:
        """Get the current face position and area.

        Returns:
            tuple: A tuple containing the face position (x, y, z) in Reachy's frame and area in pixels.
        """
        return self._position, self._area

    def _run(self) -> None:
        """Run the face tracking loop, capturing frames from the camera, detecting faces,
        and updating the Reachy's head position to look at the detected face."""
        interval = 1.0 / self._freq

        while self._enabled.is_set():
            # Capture a frame from the camera and run the YOLO model
            frame = self._reachy.cameras.teleop.get_frame()[0]
            output = self.model(frame)
            results = Detections.from_ultralytics(output[0])

            # If faces are detected, process the results
            if results.xyxy.shape[0] > 0:
                valid_idxs = np.where(results.confidence >= self._threshold)[0]

                # Select the face with the largest area and get its midpoint in Reachy's frame and its area in pixels
                if len(valid_idxs) > 0:
                    boxes = results.xyxy[valid_idxs]
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    idx = valid_idxs[np.argmax(areas)]
                    bbox = results.xyxy[idx]
                    midpoint = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    )
                    pos = self._reachy.cameras.teleop.pixel_to_world(
                        midpoint[0], midpoint[1], 1
                    )
                    self._position = pos
                    self._area = areas[np.argmax(areas)]

                    # Visualize the detection results if enabled
                    if self._visualize:
                        cv2.rectangle(
                            frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(
                            frame,
                            (int(midpoint[0]), int(midpoint[1])),
                            5,
                            (255, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            f"Area: {self._area:.2f}",
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

            # If a face position is available, make the Reachy robot look at it
            if self._position is not None:
                self._reachy.head.look_at(
                    self._position[0],
                    self._position[1],
                    self._position[2],
                    duration=interval,
                    interpolation_mode="linear",
                )

            if self._visualize:
                cv2.imshow("Face Detection", frame)
                cv2.waitKey(1)

            time.sleep(interval)


class GestureController:
    """A class to control Reachy's gestures based on face tracking information."""

    def __init__(self, reachy: ReachySDK, tracker: FaceTracker, freq: float = 5.0):
        """Initialize the GestureController.

        Args:
            reachy (ReachySDK): The Reachy robot instance.
            tracker (FaceTracker): The FaceTracker instance to get face position and area.
            freq (float): Frequency of gesture updates in Hz.
        """
        self._reachy = reachy
        self._tracker = tracker
        self._freq = freq

        self._enabled = threading.Event()

        # Parameters for waving gestures
        self._waving_duration = 8.0
        self._last_wave = time.time()
        self._is_waving = False
        self._first_wave_done = False
        self._is_up = False
        self._former_arm_joints = None

        # Parameters for moving and stopping flags
        self._moving_threshold = 1500
        self._stopping_threshold = 5000
        self.stop_flag = False

        # Default joint positions
        self.waving_joints = [-10, -10, 0, -110, 0, 0, 90]
        self.waiting_joints = [25, -10, 0, -110, 0, 0, 100]

    def start(self) -> None:
        """Start the gesture control thread."""
        self._enabled.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the gesture control thread."""
        self._enabled.clear()
        self._thread.join()

    def _run(self) -> None:
        """Run the gesture control loop, checking the face position and area,
        and performing gestures based on the face tracking information :

        - Waving when a face is detected and the area is above a threshold,
        - Moving the base when the area is below a threshold,
        - Stopping when the area exceeds a stopping threshold.
        """
        interval = 1.0 / self._freq

        while self._enabled.is_set():
            # Get the face position and area from the tracker
            pos, area = self._tracker.get_face_info()

            if pos is not None and area is not None:
                # If the area is too large, stop the robot
                if area > self._stopping_threshold:
                    self.stop_flag = True
                    self.stop()
                    break

                # If the area is below the moving threshold, get the robot closer to the person
                elif area < self._moving_threshold and not self._is_waving:
                    self._move_base(area)

                # If the time since the last wave exceeds the waving duration, start a new wave
                elif time.time() - self._last_wave >= 6.0 and not self._first_wave_done:
                    self._start_wave()
                    self._is_waving = True
                    self._first_wave_done = True
                elif (
                    time.time() - self._last_wave >= self._waving_duration + 6.0
                    and not self._is_waving
                ):
                    self._start_wave()
                    self._is_waving = True

                # Loop for the waving gesture
                elif self._is_waving:
                    if time.time() - self._last_wave <= self._waving_duration:
                        # use the yaw value of the head to adjust the yaw value of the elbow joint
                        # to follow the face position
                        goal_joints = self.waving_joints.copy()
                        head_joints = self._reachy.head.get_current_positions()
                        z_joint = head_joints[2]
                        yaw_cmd = z_joint + goal_joints[2]

                        # Clip the yaw command to avoid large movements
                        goal_joints[2] = np.clip(
                            yaw_cmd,
                            self._former_arm_joints[2] - 10,
                            self._former_arm_joints[2] + 10,
                        )

                        # Make the wave gesture by adjusting the elbow and wrist joints
                        if self._is_up:
                            goal_joints[3] += 10
                            goal_joints[5] += 20
                            self._is_up = False
                        else:
                            goal_joints[3] -= 10
                            goal_joints[5] -= 20
                            self._is_up = True

                        self._reachy.l_arm.goto(goal_joints, duration=interval)
                        self._former_arm_joints = goal_joints.copy()

                    # Making a happy antennas gesture and getting back to arm default position
                    # at the end of the waving duration
                    else:
                        self._happy_antennas()
                        self._reachy.l_arm.goto(self.waiting_joints, wait=True)
                        self._is_waving = False

            time.sleep(interval)

    def _start_wave(self) -> None:
        """Start the waving gesture by setting the arm joints to a waving position
        based on the head's current yaw and pitch angles, to be in front of the face"""
        head_joints = self._reachy.head.get_current_positions()
        y_joint = head_joints[1]
        z_joint = head_joints[2]
        y_cmd = np.clip(y_joint, -10, 10)
        z_cmd = np.clip(z_joint, -35, 35)

        goal_joints = self.waving_joints.copy()
        goal_joints[2] += z_cmd
        goal_joints[3] += y_cmd

        self._reachy.l_arm.goto(goal_joints, wait=True, duration=2)
        self._former_arm_joints = goal_joints.copy()
        self._last_wave = time.time()

    def _happy_antennas(self) -> None:
        """Make the antennas of Reachy move in a happy gesture"""
        for _ in range(5):
            reachy.head.l_antenna.goal_position = 10.0
            reachy.head.r_antenna.goal_position = -10.0
            reachy.send_goal_positions()
            time.sleep(0.1)
            reachy.head.l_antenna.goal_position = -10.0
            reachy.head.r_antenna.goal_position = 10.0
            reachy.send_goal_positions()
            time.sleep(0.1)
            reachy.head.l_antenna.goal_position = 0.0
            reachy.head.r_antenna.goal_position = 0.0
            reachy.send_goal_positions()

    def _move_base(self, area: int) -> None:
        """Move the mobile base of Reachy to get closer to the detected face based on the
        area of the face in pixels. The translation command is calculated based on the area.

        Args:
            area (int): The area of the detected face in pixels.
        """
        head_joints = self._reachy.head.get_current_positions()
        self._reachy.mobile_base.rotate_by(head_joints[2])
        self._reachy.head.rotate_by(0, 0, -head_joints[2], duration=1.5)
        self._reachy.mobile_base.translate_by(
            self._get_translation_cmd(area), 0, wait=True
        )

    def _get_translation_cmd(self, area: int) -> float:
        """Calculate the translation command for the mobile base based on the area of the detected face.
        The translation command is scaled linearly.

        Args:
            area (int): The area of the detected face in pixels.
        Returns:
            float: The x-translation command for the mobile base.
        """
        area_min = 1000
        area_max = 2500
        dx_min = 0.5
        dx_max = 0.2
        dx = (area - area_min) / (area_max - area_min) * (dx_max - dx_min) + dx_min
        return dx


if __name__ == "__main__":
    # Set up logging and argument parsing
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--thres", type=float, default=0.7)
    parser.add_argument("--freq", type=float, default=30)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    # Initialize Reachy and set up the initial posture
    reachy = ReachySDK(args.ip)
    reachy.turn_on()
    reachy.mobile_base.reset_odometry()
    reachy.head.goto_posture()
    reachy.head.neck.set_speed_limits(100)
    reachy.l_arm.goto([25, -10, 0, -110, 0, 0, 100])

    # Open and close the gripper to catch the paper fan
    reachy.l_arm.gripper.open()
    time.sleep(3)
    reachy.l_arm.gripper.close()

    # Wait for user input to start the face tracking and gesture control
    input("Press Enter to start...")

    tracker = FaceTracker(
        reachy, threshold=args.thres, freq=args.freq, visualize=args.vis
    )
    gestures = GestureController(reachy, tracker, freq=5.0)

    tracker.start()
    gestures.start()

    # Main loop to keep the program running until stopped (ctrl+C or if the face is too close)
    try:
        while not gestures.stop_flag:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.stop()
        gestures.stop()
    finally:
        reachy.turn_off_smoothly()
