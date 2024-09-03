import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import gridspec
import mmgdynamics.calibrated_vessels as cvs
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel
from mmgdynamics import step
import logging

# Define vessel model and initial values
vessel = Vessel(**cvs.kvlcc2_l64)

class maneuverNodeMPC:
    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, waypoints: list, max_steps: int, N: int):
        self.vessel = vessel
        self.state = np.copy(initial_state)
        self.dT = dT
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.target_position = self.waypoints[self.current_waypoint_index]

        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin

        self.max_steps = max_steps
        self.current_step = 0
        self.N = N  # Prediction horizon for MPC
        self.history = {'x': [], 'y': [], 'heading': []}
        
        logging.info("MPC environment initialized.")

    def dynamics(self, X, u):
        """ Compute next state using the vessel model dynamics """
        nps, delta = u
        next_state = step(
            X=X,
            vessel=self.vessel,
            dT=self.dT,
            nps=nps,
            delta=delta,
            psi=self.psi
        )
        return np.squeeze(next_state)

    def cost_function(self, U_flat):
        """ Cost function for MPC optimization """
        U = U_flat.reshape((self.N, 2))  # Reshape control inputs

        cost = 0.0
        X = np.copy(self.state)
        position = np.copy(self.position)
        psi = self.psi

        for i in range(self.N):
            # Apply control inputs and compute the cost
            nps, delta = U[i]

            # Simulate next state using vessel dynamics
            X = self.dynamics(X, [nps, delta])

            # Update heading and position
            psi += X[2] * self.dT
            position[0] += X[0] * np.cos(psi) * self.dT - X[1] * np.sin(psi) * self.dT
            position[1] += X[0] * np.sin(psi) * self.dT + X[1] * np.cos(psi) * self.dT

            # Calculate distance to target and heading error
            distance_to_target = np.linalg.norm(self.target_position - position)

            # Cost function to minimize distance to waypoint and control effort
            cost += distance_to_target**2 + 0.01 * (nps**2 + delta**2)

        return cost

    def optimize_mpc(self):
        """ Solve the MPC optimization problem """
        # Initial guess for control inputs (nps, delta) over the horizon N
        U0 = np.zeros((self.N, 2)).flatten()

        # Bounds for the control inputs (nps, delta)
        bounds = [(-1, 1), (-np.pi / 6, np.pi / 6)] * self.N  # Adjust as necessary

        # Minimize the cost function
        res = minimize(self.cost_function, U0, bounds=bounds, method='SLSQP')

        # Extract the first control action from the optimized solution
        optimal_U = res.x.reshape((self.N, 2))
        return optimal_U[0]

    def update(self):
        while self.current_step < self.max_steps:
            # Solve MPC to get optimal control inputs
            optimal_u = self.optimize_mpc()

            # Apply the optimal control action to the vessel
            self.state = self.dynamics(self.state, optimal_u)

            # Update heading and position
            self.psi += self.state[2] * self.dT
            self.position[0] += self.state[0] * np.cos(self.psi) * self.dT - self.state[1] * np.sin(self.psi) * self.dT
            self.position[1] += self.state[0] * np.sin(self.psi) * self.dT + self.state[1] * np.cos(self.psi) * self.dT

            # Record history for rendering
            self.history['x'].append(self.position[0])
            self.history['y'].append(self.position[1])
            self.history['heading'].append(self.psi)

            # Calculate distance to target
            distance_to_target = np.linalg.norm(self.target_position - self.position)

            # Check if waypoint is reached
            if distance_to_target < 32:
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                self.target_position = self.waypoints[self.current_waypoint_index]

            self.current_step += 1

        return self.history


# Parameters
dT = 0.1  # Time step
max_steps = 1000  # Maximum number of steps
N = 10  # MPC prediction horizon
initial_state = np.array([0.0, 0.0, 0.0])

# Define waypoints
waypoints = [
    np.array([250, 250]),
    np.array([500, 250]),
    np.array([500, -250]),
    np.array([250, -250])
]

# Initialize MPC-based maneuver node
node_mpc = maneuverNodeMPC(vessel, initial_state, dT, waypoints, max_steps, N)

# Run simulation
history_mpc = node_mpc.update()

# Plotting results
plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(3, 2)

# Plot trajectory and goal points
ax1 = plt.subplot(gs[0:2, 0:2])
ax1.plot(history_mpc['y'], history_mpc['x'], label='Trajectory')
for wp in waypoints:
    ax1.scatter(wp[1], wp[0], color='red')  # Waypoints
    circle = plt.Circle((wp[1], wp[0]), 32, color='r', fill=False, linestyle='--')
    ax1.add_artist(circle)
ax1.set_xlabel('Y Position')
ax1.set_ylabel('X Position')
ax1.set_title('Ship Trajectory (MPC)')
ax1.legend()

# Plot heading angle over time
ax2 = plt.subplot(gs[2, 0:2])
ax2.plot(history_mpc['heading'], label='Heading Angle')
ax2.set_xlabel('Step')
ax2.set_ylabel('Heading Angle (rad)')
ax2.set_title('Heading Angle Over Time')
ax2.legend()

plt.tight_layout()
plt.savefig("mmg_mpc_waypoints_tracking.png")
plt.show()
