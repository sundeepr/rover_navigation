#!/usr/bin/env python3
"""
MoManipVLA-style Navigation for Roomba

Implements the approach from:
"MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation"
(Wu et al., CVPR 2025)

Key idea: Use pre-trained VLA models to generate end-effector waypoints,
then use motion planning to determine base movements that make those
waypoints achievable.

For Roomba (a mobile base without an arm), we adapt this by:
1. VLA generates target pose/waypoints in the camera frame
2. Bi-level optimization plans base trajectory to reach targets
3. Motion planner enforces feasibility constraints (collision, smoothness)

Architecture:
    Camera Frame -> VLA Model -> Target Waypoints (in camera/world frame)
                                        |
                                        v
                              Bi-Level Optimizer
                              /                \
                   Upper Level:              Lower Level:
                   Base pose sampling        Trajectory optimization
                             \                /
                              v              v
                           Feasible Base Velocities
                                    |
                                    v
                              Roomba Control
"""

import numpy as np
import cv2
import time
import argparse
import threading
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math
from datetime import datetime

# Flask for web streaming
try:
    from flask import Flask, Response, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available, web streaming disabled")

# Scipy for optimization
try:
    from scipy.optimize import dual_annealing, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simplified optimization")

# Import VLA model (optional - can run in mock mode)
try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available, running in mock VLA mode")


class NavigationState(Enum):
    """Navigation state machine states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    REACHED_GOAL = "reached_goal"
    STUCK = "stuck"


@dataclass
class Waypoint:
    """A 2D waypoint for the mobile base."""
    x: float  # meters, forward from robot
    y: float  # meters, left from robot
    theta: float  # radians, heading
    confidence: float = 1.0  # VLA confidence

    def distance_to(self, other: 'Waypoint') -> float:
        """Euclidean distance to another waypoint."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: 'Waypoint') -> float:
        """Angle to another waypoint."""
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class Trajectory:
    """A trajectory is a sequence of waypoints with timing."""
    waypoints: List[Waypoint]
    timestamps: List[float]  # seconds
    feasibility_score: float = 0.0

    def __len__(self):
        return len(self.waypoints)

    def total_distance(self) -> float:
        """Total path length."""
        dist = 0.0
        for i in range(1, len(self.waypoints)):
            dist += self.waypoints[i-1].distance_to(self.waypoints[i])
        return dist

    def smoothness_cost(self) -> float:
        """Compute smoothness cost (sum of squared velocity changes)."""
        if len(self.waypoints) < 3:
            return 0.0

        cost = 0.0
        for i in range(1, len(self.waypoints) - 1):
            # Approximate acceleration as change in velocity
            dt1 = max(self.timestamps[i] - self.timestamps[i-1], 0.01)
            dt2 = max(self.timestamps[i+1] - self.timestamps[i], 0.01)

            v1_x = (self.waypoints[i].x - self.waypoints[i-1].x) / dt1
            v1_y = (self.waypoints[i].y - self.waypoints[i-1].y) / dt1
            v2_x = (self.waypoints[i+1].x - self.waypoints[i].x) / dt2
            v2_y = (self.waypoints[i+1].y - self.waypoints[i].y) / dt2

            # Squared acceleration
            cost += (v2_x - v1_x)**2 + (v2_y - v1_y)**2

        return cost


@dataclass
class RobotState:
    """Current state of the robot."""
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    theta: float = 0.0  # radians
    vx: float = 0.0  # m/s
    vy: float = 0.0  # m/s
    omega: float = 0.0  # rad/s


class VLAWaypointGenerator:
    """
    Generates target waypoints from VLA model predictions.

    The VLA model outputs end-effector poses. For a mobile base,
    we interpret these as target navigation waypoints.
    """

    def __init__(self, model_name: str = "lerobot/smolvla_base", device: str = None):
        """
        Initialize VLA model for waypoint generation.

        Args:
            model_name: HuggingFace model name
            device: torch device (auto-detect if None)
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = device

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for VLA model. Install with: pip install torch")

        self._load_model()

    def _load_model(self):
        """Load the VLA model."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading VLA model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            if "smolvla" in self.model_name.lower():
                # SmolVLA model — weights loaded to CPU in pretrained.py, then moved to device
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                self.model = SmolVLAPolicy.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_smolvla = True

                # Get tokenizer
                if hasattr(self.model, 'text_tokenizer'):
                    self.tokenizer = self.model.text_tokenizer
                else:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

                print("SmolVLA model loaded successfully")
            else:
                # OpenVLA model
                from transformers import AutoModelForVision2Seq, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_smolvla = False
                print("OpenVLA model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load VLA model '{self.model_name}': {e}")

    def generate_waypoints(
        self,
        frame: np.ndarray,
        instruction: str,
        num_waypoints: int = 5,
        horizon: float = 2.0  # meters
    ) -> List[Waypoint]:
        """
        Generate navigation waypoints from camera frame and instruction.

        This is the key adaptation from MoManipVLA:
        - VLA outputs end-effector actions
        - We interpret these as navigation targets in camera frame
        - Transform to robot-centric waypoints

        Args:
            frame: BGR camera frame
            instruction: Natural language navigation instruction
            num_waypoints: Number of waypoints to generate
            horizon: Maximum distance horizon in meters

        Returns:
            List of Waypoint objects
        """
        # Get VLA action prediction
        action_array = self._predict_vla_action(frame, instruction)

        # Interpret VLA output as navigation waypoints
        # VLA typically outputs [forward, turn, gripper, ...] style actions
        # We use the first 2-3 dimensions for navigation
        waypoints = self._action_to_waypoints(action_array, num_waypoints, horizon)

        return waypoints

    def _predict_vla_action(self, frame: np.ndarray, instruction: str) -> np.ndarray:
        """Run VLA inference to get action array."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            if self.is_smolvla:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

                image_tensor = transform(rgb_frame).unsqueeze(0).to(self.device)
                state_tensor = torch.zeros((1, 7), dtype=torch.float32).to(self.device)

                tokens = self.tokenizer(
                    instruction,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                language_tokens = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device).bool()

                observation = {
                    'observation.images.camera1': image_tensor,
                    'observation.state': state_tensor,
                    'observation.language.tokens': language_tokens,
                    'observation.language.attention_mask': attention_mask,
                }

                action = self.model.select_action(observation)
                return action.cpu().numpy().squeeze()
            else:
                pil_image = Image.fromarray(rgb_frame)
                prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

                inputs = self.processor(prompt, pil_image).to(
                    self.device,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )

                action = self.model.predict_action(**inputs, do_sample=False)
                return action.cpu().numpy().squeeze()

    def _action_to_waypoints(
        self,
        action: np.ndarray,
        num_waypoints: int,
        horizon: float
    ) -> List[Waypoint]:
        """
        Convert VLA action output to navigation waypoints.

        MoManipVLA approach: VLA outputs end-effector waypoints,
        we need to generate a trajectory that the base can follow
        to make those waypoints reachable.

        For Roomba (no arm), we interpret VLA actions as:
        - action[0]: forward/backward intent (-1 to 1)
        - action[1]: left/right turn intent (-1 to 1)
        - Higher dimensions: future waypoint hints
        """
        waypoints = []

        # SmolVLA outputs 7-DOF arm actions: [x_lateral, y_depth, z_vertical, rx, ry, rz, gripper]
        # action[1] = arm depth (reaching toward target) → forward navigation intent
        # action[0] = arm lateral → turn intent
        # Also enforce a minimum forward bias so robot always moves forward
        print(f"  [VLA raw] {[round(float(a), 3) for a in action[:6]]}")
        forward_intent = max(float(action[1]) if len(action) > 1 else 0.0, 0.3)
        turn_intent = float(action[0]) if len(action) > 0 else 0.0

        # Scale intents to physical values
        # Forward: -1 to 1 maps to -horizon to +horizon meters
        # Turn: -1 to 1 maps to -pi/2 to +pi/2 radians
        base_distance = forward_intent * horizon * 0.5  # Scale down for safety
        base_turn = turn_intent * (math.pi / 8)  # Max ~22 degree turn (reduced for smoother steering)

        # Generate waypoints along a curved path
        # This creates a smooth trajectory that respects the VLA intent
        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints

            # Interpolate along curved path
            # Use clothoid-like curve for smooth transitions
            theta_i = base_turn * t

            # Forward distance with curvature
            if abs(base_turn) > 0.01:
                # Arc length formula
                radius = abs(base_distance / base_turn) if abs(base_turn) > 0.01 else 1000
                x_i = radius * math.sin(theta_i) * np.sign(base_distance)
                y_i = radius * (1 - math.cos(theta_i)) * np.sign(base_turn)
            else:
                # Straight line
                x_i = base_distance * t
                y_i = 0.0

            # Confidence decreases with distance
            confidence = 1.0 - 0.5 * t

            waypoint = Waypoint(
                x=x_i,
                y=y_i,
                theta=theta_i,
                confidence=confidence
            )
            waypoints.append(waypoint)

        return waypoints


class MockVLAWaypointGenerator:
    """
    Mock waypoint generator for testing without a real VLA model.
    Uses simple computer vision heuristics instead of VLA inference.
    """

    def __init__(self):
        self.is_mock = True

    def generate_waypoints(
        self,
        frame: np.ndarray,
        instruction: str,
        num_waypoints: int = 5,
        horizon: float = 2.0
    ) -> List[Waypoint]:
        """Generate mock waypoints using simple CV heuristics."""
        height, width = frame.shape[:2]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Analyze left/right edge density to determine turn direction
        left_half = edges[:, :width//2]
        right_half = edges[:, width//2:]

        left_density = np.sum(left_half) / left_half.size
        right_density = np.sum(right_half) / right_half.size

        # Turn away from obstacles (more edges = more obstacles)
        turn_bias = (left_density - right_density) * 0.5
        turn_bias = np.clip(turn_bias, -0.2, 0.2)

        # Check for obstacles ahead
        center_region = edges[height//2:, width//3:2*width//3]
        obstacle_density = np.sum(center_region) / center_region.size

        # Forward speed inversely proportional to obstacle density
        forward_scale = max(0.5, 1.0 - obstacle_density * 3)

        # Parse instruction for intent
        instruction_lower = instruction.lower()
        if 'left' in instruction_lower:
            turn_bias -= 0.15
        elif 'right' in instruction_lower:
            turn_bias += 0.15
        elif 'stop' in instruction_lower:
            forward_scale = 0.0
        elif 'back' in instruction_lower:
            forward_scale = -0.3

        # Generate waypoints
        waypoints = []
        base_turn = turn_bias * (math.pi / 6)
        base_distance = forward_scale * horizon * 0.4

        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints
            theta_i = base_turn * t

            if abs(base_turn) > 0.01:
                radius = abs(base_distance / base_turn) if abs(base_turn) > 0.01 else 1000
                x_i = radius * math.sin(theta_i) * np.sign(base_distance)
                y_i = radius * (1 - math.cos(theta_i)) * np.sign(base_turn)
            else:
                x_i = base_distance * t
                y_i = 0.0

            waypoint = Waypoint(
                x=x_i,
                y=y_i,
                theta=theta_i,
                confidence=1.0 - 0.3 * t
            )
            waypoints.append(waypoint)

        return waypoints


class BiLevelMotionPlanner:
    """
    Bi-level optimization for mobile base motion planning.

    Following MoManipVLA's approach:
    - Upper level: Optimize base poses to maximize manipulability
    - Lower level: Optimize trajectory to reach waypoints

    For Roomba (no arm), we adapt:
    - Upper level: Sample candidate paths to waypoints
    - Lower level: Optimize for smoothness and collision avoidance
    """

    # Roomba physical constraints
    MAX_VELOCITY = 200  # mm/s
    MIN_VELOCITY = -200  # mm/s
    MAX_TURN_RATE = 1.0  # rad/s (approximate)
    WHEEL_BASE = 235  # mm (approximate Roomba wheel separation)

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        collision_weight: float = 10.0,
        reachability_weight: float = 5.0,
        step_size: float = 0.05  # meters between trajectory points
    ):
        """
        Initialize motion planner.

        Args:
            smoothness_weight: Weight for trajectory smoothness cost
            collision_weight: Weight for collision avoidance cost
            reachability_weight: Weight for waypoint reachability
            step_size: Distance between interpolated trajectory points
        """
        self.smoothness_weight = smoothness_weight
        self.collision_weight = collision_weight
        self.reachability_weight = reachability_weight
        self.step_size = step_size

        # Obstacle map (updated from camera)
        self.obstacle_map = None
        self.obstacle_resolution = 0.05  # meters per pixel

    def update_obstacle_map(self, frame: np.ndarray, depth_map: np.ndarray = None):
        """
        Update obstacle map from camera frame.

        For monocular camera, we use edge detection and color
        segmentation as proxy for obstacles.

        Args:
            frame: BGR camera frame
            depth_map: Optional depth image (if available)
        """
        height, width = frame.shape[:2]

        # Create simple obstacle probability map
        # This is a simplified version - real implementation would
        # use proper occupancy grid mapping

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection as obstacle indicator
        edges = cv2.Canny(gray, 50, 150)

        # Floor detection (assume floor is dominant color in bottom half)
        bottom_half = frame[height//2:, :]
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)

        # Get dominant hue as floor color
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)

        # Create floor mask
        lower = np.array([max(0, dominant_hue - 20), 30, 30])
        upper = np.array([min(179, dominant_hue + 20), 255, 255])

        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        floor_mask = cv2.inRange(hsv_full, lower, upper)

        # Combine: obstacles = edges AND NOT floor
        obstacle_prob = edges.astype(float) / 255.0
        floor_prob = floor_mask.astype(float) / 255.0

        # Obstacles are more likely where we have edges and no floor
        obstacle_prob = obstacle_prob * (1 - floor_prob * 0.5)

        # Resize to obstacle map resolution
        map_height = int(2.0 / self.obstacle_resolution)  # 2 meter range
        map_width = int(2.0 / self.obstacle_resolution)

        # Project to bird's eye view (simplified perspective transform)
        # Assume camera looking forward, obstacles at ~1m height
        self.obstacle_map = cv2.resize(obstacle_prob, (map_width, map_height))

    def plan_trajectory(
        self,
        current_state: RobotState,
        waypoints: List[Waypoint],
        time_horizon: float = 2.0
    ) -> Tuple[Trajectory, Tuple[int, int]]:
        """
        Plan trajectory to reach waypoints using bi-level optimization.

        Upper level: Sample candidate base trajectories
        Lower level: Optimize each for feasibility

        Args:
            current_state: Current robot state
            waypoints: Target waypoints from VLA
            time_horizon: Planning time horizon in seconds

        Returns:
            Tuple of (best trajectory, (left_vel, right_vel) command)
        """
        if not waypoints:
            # No waypoints - stop
            return Trajectory([], [], 0.0), (0, 0)

        # Upper level: Sample candidate trajectories
        candidates = self._sample_trajectories(current_state, waypoints, time_horizon)

        # Lower level: Evaluate and optimize each candidate
        best_trajectory = None
        best_score = float('-inf')

        for candidate in candidates:
            # Compute feasibility score
            score = self._evaluate_trajectory(candidate, waypoints)

            if score > best_score:
                best_score = score
                best_trajectory = candidate

        if best_trajectory is None or len(best_trajectory) == 0:
            # Fallback: move toward first waypoint
            return self._fallback_trajectory(current_state, waypoints[0])

        best_trajectory.feasibility_score = best_score

        # Convert first segment to wheel velocities
        velocities = self._trajectory_to_velocities(best_trajectory, current_state)

        return best_trajectory, velocities

    def _sample_trajectories(
        self,
        current_state: RobotState,
        waypoints: List[Waypoint],
        time_horizon: float,
        num_samples: int = 10
    ) -> List[Trajectory]:
        """
        Upper level: Sample candidate trajectories.

        Generate diverse trajectories that could reach the waypoints.
        """
        trajectories = []

        # Target is first high-confidence waypoint
        target = waypoints[0]
        for wp in waypoints:
            if wp.confidence > target.confidence:
                target = wp

        # Sample different approach strategies
        for i in range(num_samples):
            # Vary the curvature of approach
            curvature_bias = (i - num_samples // 2) / num_samples * 0.5

            traj_waypoints = []
            traj_times = []

            # Generate trajectory points
            num_steps = max(3, int(target.distance_to(Waypoint(0, 0, 0)) / self.step_size))

            for j in range(num_steps + 1):
                t = j / num_steps

                # Interpolate with curvature
                x = target.x * t
                y = target.y * t + curvature_bias * math.sin(math.pi * t) * target.x * 0.5
                theta = math.atan2(target.y + curvature_bias, target.x) * t

                wp = Waypoint(x, y, theta, confidence=1.0)
                traj_waypoints.append(wp)
                traj_times.append(t * time_horizon)

            trajectory = Trajectory(traj_waypoints, traj_times)
            trajectories.append(trajectory)

        return trajectories

    def _evaluate_trajectory(
        self,
        trajectory: Trajectory,
        target_waypoints: List[Waypoint]
    ) -> float:
        """
        Lower level: Evaluate trajectory feasibility.

        Computes combined score from:
        - Reachability: Does it reach the target?
        - Smoothness: Is it smooth (low acceleration)?
        - Collision: Does it avoid obstacles?
        """
        if len(trajectory) == 0:
            return float('-inf')

        # Reachability score
        final_wp = trajectory.waypoints[-1]
        target = target_waypoints[0]
        distance_error = final_wp.distance_to(target)
        reachability_score = -distance_error * self.reachability_weight

        # Smoothness score
        smoothness_cost = trajectory.smoothness_cost()
        smoothness_score = -smoothness_cost * self.smoothness_weight

        # Collision score
        collision_cost = self._compute_collision_cost(trajectory)
        collision_score = -collision_cost * self.collision_weight

        total_score = reachability_score + smoothness_score + collision_score

        return total_score

    def _compute_collision_cost(self, trajectory: Trajectory) -> float:
        """Compute collision cost using obstacle map."""
        if self.obstacle_map is None:
            return 0.0

        cost = 0.0
        map_height, map_width = self.obstacle_map.shape

        for wp in trajectory.waypoints:
            # Convert waypoint to obstacle map coordinates
            # Assume map center is robot position
            map_x = int(map_width / 2 + wp.y / self.obstacle_resolution)
            map_y = int(map_height / 2 - wp.x / self.obstacle_resolution)

            # Check bounds
            if 0 <= map_x < map_width and 0 <= map_y < map_height:
                cost += self.obstacle_map[map_y, map_x]

        return cost / max(len(trajectory), 1)

    def _trajectory_to_velocities(
        self,
        trajectory: Trajectory,
        current_state: RobotState
    ) -> Tuple[int, int]:
        """
        Convert trajectory to Roomba wheel velocities.

        Uses differential drive kinematics:
        v = (v_r + v_l) / 2
        omega = (v_r - v_l) / wheel_base
        """
        if len(trajectory) < 2:
            return (0, 0)

        # Get desired motion to next waypoint
        next_wp = trajectory.waypoints[1]
        dt = max(trajectory.timestamps[1] - trajectory.timestamps[0], 0.1)

        # Desired linear and angular velocity
        dx = next_wp.x - current_state.x
        dy = next_wp.y - current_state.y
        dtheta = next_wp.theta - current_state.theta

        # Normalize angle
        while dtheta > math.pi:
            dtheta -= 2 * math.pi
        while dtheta < -math.pi:
            dtheta += 2 * math.pi

        # Compute velocities
        linear_vel = math.sqrt(dx**2 + dy**2) / dt  # m/s
        angular_vel = dtheta / dt  # rad/s

        # Convert to mm/s
        linear_vel_mm = linear_vel * 1000

        # Differential drive inverse kinematics
        # v = (v_r + v_l) / 2
        # omega = (v_r - v_l) / L
        # Therefore:
        # v_r = v + omega * L / 2
        # v_l = v - omega * L / 2

        v_r = linear_vel_mm + angular_vel * self.WHEEL_BASE / 2
        v_l = linear_vel_mm - angular_vel * self.WHEEL_BASE / 2

        # Clamp to Roomba limits
        v_r = int(np.clip(v_r, self.MIN_VELOCITY, self.MAX_VELOCITY))
        v_l = int(np.clip(v_l, self.MIN_VELOCITY, self.MAX_VELOCITY))

        # Ensure minimum movement if velocities are too small but non-zero intent
        min_vel = 50  # Minimum velocity to overcome friction (reduced)
        if abs(v_r) < min_vel and abs(v_l) < min_vel:
            if linear_vel_mm > 10:  # If there's forward intent
                scale = min_vel / max(abs(v_r), abs(v_l), 1)
                v_r = int(v_r * scale) if v_r != 0 else min_vel
                v_l = int(v_l * scale) if v_l != 0 else min_vel

        return (v_l, v_r)

    def _fallback_trajectory(
        self,
        current_state: RobotState,
        target: Waypoint
    ) -> Tuple[Trajectory, Tuple[int, int]]:
        """Fallback simple trajectory to target."""
        # Simple proportional control toward target
        distance = math.sqrt(target.x**2 + target.y**2)
        angle = math.atan2(target.y, target.x)

        # Limit angle to avoid spinning in place
        angle = np.clip(angle, -0.5, 0.5)  # Max ~30 degrees

        # Base forward velocity - ensure robot moves forward
        base_vel = 100  # mm/s base speed

        # Linear velocity - always move forward
        linear_vel = base_vel

        # Angular velocity proportional to angle error (reduced gain)
        angular_vel = angle * 0.5  # rad/s (reduced to favor forward motion)

        # Differential drive
        v_r = linear_vel + angular_vel * self.WHEEL_BASE / 2
        v_l = linear_vel - angular_vel * self.WHEEL_BASE / 2

        # Ensure both wheels move forward (no spinning in place)
        v_r = int(np.clip(v_r, 50, self.MAX_VELOCITY))
        v_l = int(np.clip(v_l, 50, self.MAX_VELOCITY))

        # Create simple trajectory
        wp = Waypoint(target.x * 0.1, target.y * 0.1, angle * 0.1)
        traj = Trajectory([Waypoint(0, 0, 0), wp], [0.0, 0.1], feasibility_score=0.5)

        return traj, (v_l, v_r)


class MoManipNavigator:
    """
    Main MoManipVLA-style navigation controller for Roomba.

    Integrates:
    - VLA waypoint generation
    - Bi-level motion planning
    - Real-time control loop
    """

    def __init__(
        self,
        vla_model: str = "lerobot/smolvla_base",
        use_mock_vla: bool = False,
        planning_frequency: float = 5.0,  # Hz
        control_frequency: float = 10.0,  # Hz
    ):
        """
        Initialize MoManip navigator.

        Args:
            vla_model: VLA model name
            use_mock_vla: Use mock VLA (for testing without model)
            planning_frequency: How often to replan (Hz)
            control_frequency: How often to send commands (Hz)
        """
        self.planning_frequency = planning_frequency
        self.control_frequency = control_frequency

        # Initialize VLA waypoint generator
        if use_mock_vla:
            self.waypoint_generator = MockVLAWaypointGenerator()
        else:
            self.waypoint_generator = VLAWaypointGenerator(model_name=vla_model)

        # Initialize motion planner
        self.motion_planner = BiLevelMotionPlanner()

        # State
        self.state = NavigationState.IDLE
        self.current_state = RobotState()
        self.current_waypoints: List[Waypoint] = []
        self.current_trajectory: Optional[Trajectory] = None
        self.current_velocities = (0, 0)

        # Instruction
        self.instruction = "navigate forward to the green ball"

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'waypoints_generated': 0,
            'trajectories_planned': 0,
            'planning_time_ms': 0,
            'vla_inference_time_ms': 0,
        }

        # Thread safety
        self.lock = threading.Lock()
        self._running = False

    def process_frame(
        self,
        frame: np.ndarray,
        instruction: str = None
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Process a camera frame and return wheel velocities.

        This is the main entry point for the navigation system.

        Args:
            frame: BGR camera frame
            instruction: Optional navigation instruction

        Returns:
            Tuple of (left_vel, right_vel, debug_info)
        """
        if instruction is not None:
            self.instruction = instruction

        with self.lock:
            self.stats['frames_processed'] += 1

            # Step 1: Generate waypoints from VLA
            vla_start = time.time()
            waypoints = self.waypoint_generator.generate_waypoints(
                frame,
                self.instruction,
                num_waypoints=5,
                horizon=1.5
            )
            vla_time = (time.time() - vla_start) * 1000
            self.stats['vla_inference_time_ms'] = vla_time
            self.stats['waypoints_generated'] = len(waypoints)

            self.current_waypoints = waypoints

            # Step 2: Update obstacle map
            self.motion_planner.update_obstacle_map(frame)

            # Step 3: Plan trajectory using bi-level optimization
            plan_start = time.time()
            trajectory, velocities = self.motion_planner.plan_trajectory(
                self.current_state,
                waypoints,
                time_horizon=2.0
            )
            plan_time = (time.time() - plan_start) * 1000
            self.stats['planning_time_ms'] = plan_time
            self.stats['trajectories_planned'] += 1

            self.current_trajectory = trajectory
            self.current_velocities = velocities

            # Build debug info
            debug_info = {
                'waypoints': [(wp.x, wp.y, wp.theta, wp.confidence) for wp in waypoints],
                'waypoints_generated': len(waypoints),
                'trajectory_length': len(trajectory) if trajectory else 0,
                'trajectory_score': trajectory.feasibility_score if trajectory else 0,
                'vla_time_ms': vla_time,
                'planning_time_ms': plan_time,
                'total_time_ms': vla_time + plan_time,
                'instruction': self.instruction,
            }

            return velocities[0], velocities[1], debug_info

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw visualization overlays on frame.

        Shows:
        - Waypoints as circles
        - Planned trajectory as line
        - Current velocities
        - Statistics
        """
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]

        # Draw waypoints (project from robot frame to image)
        # Simplified: assume camera forward is image center-bottom
        for i, wp in enumerate(self.current_waypoints):
            # Scale waypoint position to image coordinates
            # x (forward) maps to vertical (up), y (left) maps to horizontal
            img_x = int(width / 2 - wp.y * 200)  # 200 pixels per meter
            img_y = int(height - wp.x * 200)

            # Clamp to frame
            img_x = np.clip(img_x, 0, width - 1)
            img_y = np.clip(img_y, 0, height - 1)

            # Color based on confidence
            color = (
                int(255 * (1 - wp.confidence)),
                int(255 * wp.confidence),
                0
            )
            radius = int(10 * wp.confidence) + 5

            cv2.circle(vis_frame, (img_x, img_y), radius, color, -1)
            cv2.putText(
                vis_frame, f"{i+1}",
                (img_x - 5, img_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Draw trajectory
        if self.current_trajectory and len(self.current_trajectory) > 1:
            points = []
            for wp in self.current_trajectory.waypoints:
                img_x = int(width / 2 - wp.y * 200)
                img_y = int(height - wp.x * 200)
                img_x = np.clip(img_x, 0, width - 1)
                img_y = np.clip(img_y, 0, height - 1)
                points.append((img_x, img_y))

            for i in range(len(points) - 1):
                cv2.line(vis_frame, points[i], points[i+1], (255, 0, 255), 2)

        # Draw velocity indicator
        left_vel, right_vel = self.current_velocities

        # Velocity bars
        bar_height = 100
        bar_width = 20
        bar_x_left = 30
        bar_x_right = width - 50
        bar_y = height - 150

        # Left wheel
        left_bar_height = int(abs(left_vel) / 500 * bar_height)
        left_color = (0, 255, 0) if left_vel >= 0 else (0, 0, 255)
        cv2.rectangle(
            vis_frame,
            (bar_x_left, bar_y + bar_height - left_bar_height),
            (bar_x_left + bar_width, bar_y + bar_height),
            left_color, -1
        )
        cv2.putText(
            vis_frame, f"L:{left_vel}",
            (bar_x_left - 5, bar_y + bar_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Right wheel
        right_bar_height = int(abs(right_vel) / 500 * bar_height)
        right_color = (0, 255, 0) if right_vel >= 0 else (0, 0, 255)
        cv2.rectangle(
            vis_frame,
            (bar_x_right, bar_y + bar_height - right_bar_height),
            (bar_x_right + bar_width, bar_y + bar_height),
            right_color, -1
        )
        cv2.putText(
            vis_frame, f"R:{right_vel}",
            (bar_x_right - 5, bar_y + bar_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Statistics overlay
        y_offset = 30
        stats_text = [
            f"MoManipVLA Navigation",
            f"Instruction: {self.instruction}",
            f"Waypoints: {len(self.current_waypoints)}",
            f"Traj Score: {self.current_trajectory.feasibility_score:.2f}" if self.current_trajectory else "Traj Score: N/A",
            f"VLA: {self.stats['vla_inference_time_ms']:.0f}ms",
            f"Plan: {self.stats['planning_time_ms']:.0f}ms",
            f"Frame: {self.stats['frames_processed']}",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(
                vis_frame, text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                vis_frame, text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1
            )

        return vis_frame


# ============================================================================
# Web Streaming Server
# ============================================================================

# Global variables for Flask streaming
web_app = Flask(__name__) if FLASK_AVAILABLE else None
web_output_frame = None
web_frame_lock = threading.Lock()
web_nav_stats = {
    'frames_processed': 0,
    'waypoints': 0,
    'trajectory_score': 0.0,
    'left_vel': 0,
    'right_vel': 0,
    'vla_time_ms': 0,
    'planning_time_ms': 0,
    'instruction': '',
    'paused': False,
    'running': True,
}

# HTML template for web dashboard
WEB_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MoManipVLA Navigation</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            background-color: #0a0a0a;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #00d4ff;
            border-radius: 8px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .button {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000;
            border: none;
            padding: 12px 24px;
            margin: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        }
        .button.stop {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: #fff;
        }
        .button.stop:hover {
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.4);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00d4ff;
            backdrop-filter: blur(10px);
        }
        .stat-label {
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
            color: #00d4ff;
        }
        .status {
            text-align: center;
            padding: 12px 24px;
            border-radius: 8px;
            margin: 15px auto;
            max-width: 200px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .status.running {
            background: rgba(0, 255, 100, 0.2);
            color: #00ff64;
            border: 1px solid #00ff64;
        }
        .status.paused {
            background: rgba(255, 200, 0, 0.2);
            color: #ffc800;
            border: 1px solid #ffc800;
        }
        .instruction-box {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
        }
        .instruction-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .instruction-text {
            color: #00d4ff;
            font-size: 18px;
            font-style: italic;
        }
        .velocity-display {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
        }
        .wheel {
            text-align: center;
        }
        .wheel-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .wheel-value {
            font-size: 36px;
            font-weight: bold;
        }
        .wheel-value.positive { color: #00ff64; }
        .wheel-value.negative { color: #ff4444; }
        .wheel-value.zero { color: #888; }
    </style>
    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('frame_count').textContent = data.frames_processed;
                    document.getElementById('waypoints').textContent = data.waypoints;
                    document.getElementById('traj_score').textContent = data.trajectory_score.toFixed(2);
                    document.getElementById('vla_time').textContent = data.vla_time_ms.toFixed(0) + 'ms';
                    document.getElementById('plan_time').textContent = data.planning_time_ms.toFixed(0) + 'ms';

                    // Update velocities
                    const leftVel = document.getElementById('left_vel');
                    const rightVel = document.getElementById('right_vel');
                    leftVel.textContent = data.left_vel;
                    rightVel.textContent = data.right_vel;

                    leftVel.className = 'wheel-value ' + (data.left_vel > 0 ? 'positive' : (data.left_vel < 0 ? 'negative' : 'zero'));
                    rightVel.className = 'wheel-value ' + (data.right_vel > 0 ? 'positive' : (data.right_vel < 0 ? 'negative' : 'zero'));

                    // Update instruction
                    document.getElementById('instruction').textContent = '"' + data.instruction + '"';

                    // Update status
                    const statusDiv = document.getElementById('status');
                    if (data.paused) {
                        statusDiv.className = 'status paused';
                        statusDiv.textContent = 'PAUSED';
                    } else {
                        statusDiv.className = 'status running';
                        statusDiv.textContent = 'RUNNING';
                    }
                });
        }

        function togglePause() {
            fetch('/toggle_pause').then(response => response.json());
        }

        function emergencyStop() {
            if (confirm('Emergency stop navigation?')) {
                fetch('/emergency_stop').then(() => {
                    alert('Navigation stopped!');
                });
            }
        }

        // Update stats every 500ms
        setInterval(updateStats, 500);
        updateStats();
    </script>
</head>
<body>
    <div class="container">
        <h1>MoManipVLA Navigation</h1>
        <div class="subtitle">Vision-Language-Action Model for Mobile Base Control</div>

        <div id="status" class="status running">RUNNING</div>

        <div class="instruction-box">
            <div class="instruction-label">Current Instruction</div>
            <div class="instruction-text" id="instruction">"navigate forward"</div>
        </div>

        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Navigation Feed">
        </div>

        <div class="velocity-display">
            <div class="wheel">
                <div class="wheel-label">LEFT WHEEL</div>
                <div class="wheel-value zero" id="left_vel">0</div>
                <div class="wheel-label">mm/s</div>
            </div>
            <div class="wheel">
                <div class="wheel-label">RIGHT WHEEL</div>
                <div class="wheel-value zero" id="right_vel">0</div>
                <div class="wheel-label">mm/s</div>
            </div>
        </div>

        <div class="controls">
            <button class="button" onclick="togglePause()">Pause / Resume</button>
            <button class="button stop" onclick="emergencyStop()">Emergency Stop</button>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Frames</div>
                <div class="stat-value" id="frame_count">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Waypoints</div>
                <div class="stat-value" id="waypoints">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Traj Score</div>
                <div class="stat-value" id="traj_score">0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">VLA Time</div>
                <div class="stat-value" id="vla_time">0ms</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Plan Time</div>
                <div class="stat-value" id="plan_time">0ms</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

if FLASK_AVAILABLE:
    @web_app.route('/')
    def web_index():
        """Serve the main dashboard page."""
        return render_template_string(WEB_TEMPLATE)

    @web_app.route('/video_feed')
    def video_feed():
        """Video streaming route."""
        def generate():
            global web_output_frame, web_frame_lock
            while True:
                with web_frame_lock:
                    if web_output_frame is None:
                        # Send a blank frame if no frame available
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "Waiting for frames...", (180, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                        ret, buffer = cv2.imencode('.jpg', blank)
                    else:
                        ret, buffer = cv2.imencode('.jpg', web_output_frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.05)  # ~20 FPS

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @web_app.route('/stats')
    def web_stats():
        """Return navigation statistics as JSON."""
        global web_nav_stats
        return jsonify(web_nav_stats)

    @web_app.route('/toggle_pause')
    def toggle_pause():
        """Toggle pause state."""
        global web_nav_stats
        web_nav_stats['paused'] = not web_nav_stats['paused']
        return jsonify({'paused': web_nav_stats['paused']})

    @web_app.route('/emergency_stop')
    def emergency_stop():
        """Emergency stop."""
        global web_nav_stats
        web_nav_stats['running'] = False
        web_nav_stats['paused'] = True
        return jsonify({'stopped': True})


def start_web_server(port: int = 5001):
    """Start Flask web server in background thread."""
    if not FLASK_AVAILABLE:
        print("Warning: Flask not available, web streaming disabled")
        return None

    def run_server():
        # Suppress Flask startup messages
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        web_app.run(host='0.0.0.0', port=port, debug=False,
                   use_reloader=False, threaded=True)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread


def update_web_frame(frame: np.ndarray, stats: Dict[str, Any]):
    """Update the web streaming frame and stats."""
    global web_output_frame, web_frame_lock, web_nav_stats

    with web_frame_lock:
        web_output_frame = frame.copy()

    web_nav_stats.update({
        'frames_processed': stats.get('frames_processed', 0),
        'waypoints': stats.get('waypoints_generated', 0),
        'trajectory_score': stats.get('trajectory_score', 0.0),
        'left_vel': stats.get('left_vel', 0),
        'right_vel': stats.get('right_vel', 0),
        'vla_time_ms': stats.get('vla_time_ms', 0),
        'planning_time_ms': stats.get('planning_time_ms', 0),
        'instruction': stats.get('instruction', ''),
    })


# ============================================================================
# Utility Functions
# ============================================================================

def check_display_available():
    """Check if display/GUI is available."""
    try:
        # Try to create a small test window
        test_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imshow('_test_', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('_test_')
        return True
    except cv2.error:
        return False


def main():
    """Main entry point for MoManipVLA navigation."""
    parser = argparse.ArgumentParser(
        description='MoManipVLA-style Navigation for Roomba',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with mock VLA (no model required)
  python momanip_navigation.py --mock-vla --camera 0

  # Run with SmolVLA model
  python momanip_navigation.py --model lerobot/smolvla_base --camera 0

  # Dry run (no Roomba connection)
  python momanip_navigation.py --mock-vla --dry-run

  # Custom instruction
  python momanip_navigation.py --mock-vla --instruction "turn left and go forward"
        """
    )

    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--model', default='lerobot/smolvla_base',
                       help='VLA model name')
    parser.add_argument('--mock-vla', action='store_true',
                       help='Use mock VLA (no model loading)')
    parser.add_argument('--instruction', default='navigate to the yellow box',
                       help='Navigation instruction')
    parser.add_argument('--port', default='/dev/ttyUSB0',
                       help='Roomba serial port')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (no Roomba commands)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visualization window')
    parser.add_argument('--web-port', type=int, default=5001,
                       help='Web streaming port')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save video to file (e.g., output.mp4 or output.avi)')
    parser.add_argument('--video-fps', type=float, default=20.0,
                       help='Video recording FPS (default: 20)')

    args = parser.parse_args()

    # Check if display is available
    display_available = not args.no_display and check_display_available()
    if not display_available and not args.no_display:
        print("Note: Display not available (headless mode). Use --no-display to suppress this message.")
        args.no_display = True

    print("=" * 60)
    print("MoManipVLA Navigation for Roomba")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Model: {args.model}")
    print(f"Mock VLA: {args.mock_vla}")
    print(f"Instruction: {args.instruction}")
    print(f"Dry run: {args.dry_run}")
    print(f"Display: {'enabled' if not args.no_display else 'disabled (headless)'}")
    print(f"Web UI: http://localhost:{args.web_port}")
    if args.save_video:
        print(f"Recording video to: {args.save_video} @ {args.video_fps} FPS")
    print("=" * 60)

    # Start web streaming server
    start_web_server(args.web_port)
    print(f"\nWeb dashboard available at: http://localhost:{args.web_port}")

    # Initialize navigator
    navigator = MoManipNavigator(
        vla_model=args.model,
        use_mock_vla=args.mock_vla
    )

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return 1

    # Initialize video writer (if saving video)
    video_writer = None
    video_path = None
    if args.save_video:
        # Get frame dimensions from camera
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Add timestamp to video filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = args.save_video

        # Split extension and insert timestamp
        if '.' in base_path:
            name, ext = base_path.rsplit('.', 1)
            video_path = f"{name}_{timestamp}.{ext}"
        else:
            video_path = f"{base_path}_{timestamp}.mp4"

        # Determine codec based on file extension
        if video_path.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif video_path.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif video_path.endswith('.mkv'):
            fourcc = cv2.VideoWriter_fourcc(*'X264')
        else:
            # Default to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = video_path + '.mp4'

        video_writer = cv2.VideoWriter(
            video_path, fourcc, args.video_fps, (frame_width, frame_height)
        )
        if video_writer.isOpened():
            print(f"Video writer initialized: {frame_width}x{frame_height} @ {args.video_fps} FPS")
            print(f"Recording to: {video_path}")
        else:
            print(f"Warning: Could not initialize video writer for {video_path}")
            video_writer = None

    # Initialize Roomba (if not dry run)
    roomba = None
    roomba_ctx = None
    if not args.dry_run:
        try:
            import sys
            sys.path.insert(0, '..')
            from roomba_control import Roomba
            roomba = Roomba(args.port)
            roomba_ctx = roomba.connect()
            roomba_ctx.__enter__()
            print(f"Connected to Roomba on {args.port}")

            # IMPORTANT: Send START and SAFE commands to enable motor control
            print("Sending START command...")
            roomba.start()
            time.sleep(0.1)
            print("Sending SAFE command...")
            roomba.safe()
            time.sleep(0.1)
            print("Roomba ready for control!")

        except Exception as e:
            print(f"Warning: Could not connect to Roomba: {e}")
            print("Running in dry-run mode")
            args.dry_run = True
            roomba = None

    print("\nStarting navigation loop...")
    print("Press 'q' to quit, 'p' to pause")
    print("=" * 60)

    paused = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            # Check for web UI pause/stop commands
            if web_nav_stats['paused'] and not paused:
                paused = True
                if roomba and not args.dry_run:
                    roomba.drive(0, 0x8000)
                print("PAUSED (from web UI)")
            elif not web_nav_stats['paused'] and paused:
                paused = False
                print("RESUMED (from web UI)")

            if not web_nav_stats['running']:
                print("Stopped from web UI")
                break

            if not paused:
                # Step-based control: infer once, stop, execute for fixed duration, repeat
                EXECUTE_DURATION = 0.5  # seconds to drive before re-inferring

                # Stop Roomba before inference
                if roomba and not args.dry_run:
                    roomba.drive(0, 0x8000)

                # Run VLA inference on current frame
                left_vel, right_vel, debug_info = navigator.process_frame(
                    frame,
                    args.instruction
                )

                # Send to Roomba and drive for EXECUTE_DURATION seconds
                if roomba and not args.dry_run:
                    velocity = (left_vel + right_vel) // 2
                    diff = right_vel - left_vel
                    if abs(diff) < 5:
                        radius = 0x8000  # Straight
                    elif abs(velocity) < 5:
                        radius = 1 if diff > 0 else -1
                        velocity = abs(diff) // 2
                    else:
                        wheel_base = 235  # mm
                        radius = int(wheel_base * (right_vel + left_vel) / (2 * diff))
                        radius = max(-2000, min(2000, radius))

                    print(f"  -> Drive: velocity={velocity}, radius={radius} for {EXECUTE_DURATION}s")
                    roomba.drive(velocity, radius)
                    time.sleep(EXECUTE_DURATION)
                    roomba.drive(0, 0x8000)  # Stop after execution
                elif not roomba:
                    if navigator.stats['frames_processed'] == 1:
                        print("WARNING: Roomba not connected! Commands not being sent.")
                elif args.dry_run:
                    if navigator.stats['frames_processed'] == 1:
                        print("INFO: Dry-run mode - commands not sent to Roomba")

                # Print status periodically
                if navigator.stats['frames_processed'] % 30 == 0:
                    print(f"Frame {navigator.stats['frames_processed']}: "
                          f"L={left_vel:4d} R={right_vel:4d} | "
                          f"WP={debug_info['waypoints_generated']} | "
                          f"Score={debug_info['trajectory_score']:.2f} | "
                          f"Time={debug_info['total_time_ms']:.0f}ms")
            else:
                left_vel, right_vel = 0, 0
                debug_info = {
                    'waypoints_generated': 0,
                    'trajectory_score': 0,
                    'vla_time_ms': 0,
                    'planning_time_ms': 0,
                    'instruction': args.instruction,
                }

            # Generate visualization frame (always, for web streaming)
            vis_frame = navigator.visualize(frame)

            if paused:
                cv2.putText(
                    vis_frame, "PAUSED",
                    (vis_frame.shape[1] // 2 - 80, vis_frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
                )

            # Update web streaming
            web_stats = {
                'frames_processed': navigator.stats['frames_processed'],
                'waypoints_generated': debug_info.get('waypoints_generated', 0),
                'trajectory_score': debug_info.get('trajectory_score', 0),
                'left_vel': left_vel,
                'right_vel': right_vel,
                'vla_time_ms': debug_info.get('vla_time_ms', 0),
                'planning_time_ms': debug_info.get('planning_time_ms', 0),
                'instruction': args.instruction,
            }
            update_web_frame(vis_frame, web_stats)

            # Write frame to video file
            if video_writer is not None:
                video_writer.write(vis_frame)

            # Local display (if available)
            if not args.no_display:
                try:
                    cv2.imshow('MoManipVLA Navigation', vis_frame)
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        web_nav_stats['paused'] = paused
                        if paused and roomba:
                            roomba.drive(0, 0x8000)  # Stop
                        print("PAUSED" if paused else "RESUMED")
                except cv2.error:
                    # GUI not available, switch to headless
                    args.no_display = True
                    print("Display unavailable, switching to headless mode")
            else:
                # Headless mode - add small delay to control loop rate
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        # Cleanup
        if roomba and not args.dry_run:
            roomba.drive(0, 0x8000)  # Stop
            roomba_ctx.__exit__(None, None, None)

        cap.release()

        # Release video writer
        if video_writer is not None:
            video_writer.release()
            print(f"\nVideo saved to: {video_path}")

        # Only try to destroy windows if display was available
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore if GUI not available

        print("\n" + "=" * 60)
        print("Navigation stopped")
        print(f"Total frames: {navigator.stats['frames_processed']}")
        print(f"Total trajectories: {navigator.stats['trajectories_planned']}")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
