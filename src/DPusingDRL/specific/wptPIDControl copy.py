import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel, InitialValues
from mmgdynamics import step
import logging
from scipy.optimize import curve_fit

# 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 패밀리
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기

@dataclass
class KVLCC2Inits:
    l_64 = InitialValues(
        u=0.0,  # Longitudinal vessel speed [m/s]
        v=0.0,  # Lateral vessel speed [m/s]
        r=0.0,  # Yaw rate acceleration [rad/s]
        delta=0.0,  # Rudder angle [rad]
        nps=0.0  # Propeller revs [s⁻¹]
    )

# Use a pre-calibrated vessel
vessel = Vessel(**cvs.kvlcc2_l64)
ivs = KVLCC2Inits.l_64

# 초기 상태 정의
initial_state = np.array([ivs.u, ivs.v, ivs.r])
print("Initial state set.")

# 경유점 정의
waypoints = [
    np.array([-900, 0])
    # np.array([-1200, -300]),
    # np.array([-900, -600])
]
print("Waypoints set.")

class maneuverNode:
    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, waypoints: list, max_steps: int):
        self.vessel = vessel
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.target_position = self.waypoints[self.current_waypoint_index]

        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([-850.0, -750.0])  # Assume initial position is at the origin

        self.fl_psi = 0.0
        self.fl_vel = 0.0
        self.w_vel = 0.0
        self.beta_w = 0.0

        self.max_steps = max_steps
        self.current_step = 0
        self.history = {'x': [], 'y': [], 'heading': [], 'rudder': [], 'nps': [], 'surge_speed': []}
        logging.info("Environment initialized.")

    def PD_controller(self, error_psi, error_u, Kp_psi, Kd_psi, Kp_u, Kd_u):
        delta = Kp_psi * error_psi - Kd_psi * self.state[2]
        nps = Kp_u * error_u - Kd_u * self.state[0]
        return delta, nps

    def update(self, Kp_psi, Kd_psi, Kp_u, Kd_u):
        while self.current_step < self.max_steps:
            # Calculate distance and heading to target
            distance_to_target = np.linalg.norm(self.target_position - self.position)
            heading_error = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0]) - self.psi

            desired_u = 4.0  # Desired forward speed [m/s]
            error_psi = heading_error
            error_u = desired_u - self.state[0]

            # Get control inputs
            delta, nps = self.PD_controller(error_psi, error_u, Kp_psi, Kd_psi, Kp_u, Kd_u)

            # Step the simulation
            next_state = step(
                X=self.state,
                vessel=self.vessel,
                dT=self.dT,
                nps=nps,
                delta=delta,
                psi=self.psi,
                fl_psi=self.fl_psi,
                fl_vel=self.fl_vel,
                w_vel=self.w_vel,
                beta_w=self.beta_w
            )
            self.state = np.squeeze(next_state)

            # Update heading
            self.psi += self.state[2] * self.dT

            if self.psi > np.pi:
                self.psi -= 2 * np.pi
            elif self.psi < -np.pi:
                self.psi += 2 * np.pi

            # Update position
            self.position[0] += self.state[0] * np.cos(self.psi) * self.dT - self.state[1] * np.sin(self.psi) * self.dT
            self.position[1] += self.state[0] * np.sin(self.psi) * self.dT + self.state[1] * np.cos(self.psi) * self.dT

            # Record history for rendering
            self.history['x'].append(self.position[0])
            self.history['y'].append(self.position[1])
            self.history['heading'].append(self.psi)
            self.history['rudder'].append(delta)
            self.history['nps'].append(nps)
            self.history['surge_speed'].append(self.state[0])  # 전진 속도 기록

            # Check if waypoint is reached
            if distance_to_target < 32:
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                self.target_position = self.waypoints[self.current_waypoint_index]

            self.current_step += 1

        return self.history

# Parameters
dT = 0.1  # Time step
max_steps = 2000  # Maximum number of steps

# PD controller gains
Kp_psi = 1.0
Kd_psi = 0.9
Kp_u = 1.0
Kd_u = 0.1

# Initialize maneuver node
node = maneuverNode(vessel, initial_state, dT, waypoints, max_steps)

# Run simulation
history = node.update(Kp_psi, Kd_psi, Kp_u, Kd_u)

# Plotting results
plt.figure(figsize=(5, 8))
gs = gridspec.GridSpec(3, 2)

# Plot trajectory and goal points
ax1 = plt.subplot(gs[0:2, 0:2])
ax1.plot(history['y'], history['x'], label='Trajectory')
for wp in waypoints:
    ax1.scatter(wp[1], wp[0], color='red')  # 목표 위치 표시
    circle = plt.Circle((wp[1], wp[0]), 32, color='r', fill=False, linestyle='--')
    ax1.add_artist(circle)
ax1.set_xlabel('Y Position')
ax1.set_ylabel('X Position')
ax1.set_title('Ship trajectory')
ax1.legend()

# Plot heading angle over time
ax2 = plt.subplot(gs[2, 0:2])
ax3 = ax2.twinx()
ax2.plot(np.rad2deg(history['heading']), label='psi', color='blue')
ax3.plot(np.rad2deg(history['rudder']), label='delta', color='green')
ax2.set_xlabel('t')
ax2.set_ylabel('angle (deg)')
ax2.set_title('psi vs. delta')
ax2.legend()

plt.tight_layout()
plt.savefig("mmg_waypoints_tracking.png")
plt.show()

