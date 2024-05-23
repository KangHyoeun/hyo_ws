#!/usr/bin/env python3
# wptEnv.py

import gymnasium as gym 
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from scipy.stats import multivariate_normal

# 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 패밀리
matplotlib.rcParams['font.size'] = 12  # 기본 폰트 크기

from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel
from mmgdynamics import step
import math
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ShipEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, waypoints: list, render_mode='human', max_steps=500):
        super(ShipEnv, self).__init__()
        self.vessel = vessel
        self.render_mode = render_mode
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.target_position = self.waypoints[self.current_waypoint_index]

        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin

        self.fl_psi = 0.0
        self.fl_vel = 0.0
        self.w_vel = 0.0
        self.beta_w = 0.0

        self.max_steps = max_steps
        self.current_step = 0
        self.history = {'x': [], 'y': [], 'rewards': [], '\psi': [], '\delta': []}
        logging.info("Environment initialized.")

        # Action space: [propeller revs per second(nps), rudder angle(delta)] # Normalize action space to [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)

        # Actual action bounds
        self.nps_bounds = (0.0, 13.4)
        self.delta_bounds = (np.deg2rad(-35), np.deg2rad(35))

        # Observation space: [x, y, u, v, r, distance to target, heading to target, heading error]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

    def step(self, action):
        logging.info(f"Step action received: {action}")
        self.current_step += 1

        # Rescale action from [-1, 1] to actual action bounds
        nps = (action[0] * (self.nps_bounds[1] - self.nps_bounds[0]) / 2.0 + (self.nps_bounds[1] + self.nps_bounds[0]) / 2.0)
        delta = (action[1] * (self.delta_bounds[1] - self.delta_bounds[0]) / 2.0 + (self.delta_bounds[1] + self.delta_bounds[0]) / 2.0)

        logging.info(f"nps: {nps}, delta: {delta}")

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

        # Calculate distance and heading to target
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_error = np.arctan2(self.target_position[1] - self.position[1],
                                   self.target_position[0] - self.position[0]) - self.psi
        
        if heading_error > np.pi:
            heading_error -= np.pi*2
        elif heading_error < -np.pi:
            heading_error += np.pi*2

        # Observation
        observation = np.array([self.position[0], self.position[1], self.state[0], self.state[1], self.state[2], distance_to_target, heading_error])

        # Reward
        reward = 0

        # 1. 목표 위치에 도달하는 것에 대한 보상 증가
        if distance_to_target < 32.0:
            reward += 1000.0
        elif distance_to_target < 64.0 and distance_to_target > 32.0:
            reward += 300.0
        elif distance_to_target < 128.0 and distance_to_target > 64.0:
            reward += 100.0 

        if abs(heading_error) < np.deg2rad(5):
            reward += 1000.0
        elif abs(heading_error) < np.deg2rad(10) and abs(heading_error) > np.deg2rad(5):
            reward += 300.0

        # 2. 목표 위치와의 거리에 따른 보상
        reward -= distance_to_target*0.1

        # 3. 방위각 오차에 따른 보상
        reward -= abs(heading_error)
        
        
        # 다변량 가우시안 보상 함수
        # reward = self.multivariate_gaussian_reward(distance_to_target, heading_error)
        
        # Done condition
        terminated = distance_to_target < 64.0  # Consider terminated if within 1 meter and 5 degrees of target
        truncated = self.current_step >= self.max_steps

        done = terminated or truncated  # Combine conditions for done

        if terminated and self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
            self.target_position = self.waypoints[self.current_waypoint_index]
            done = False

        # Record history for rendering
        self.history['x'].append(self.position[0])
        self.history['y'].append(self.position[1])
        self.history['rewards'].append(reward)
        self.history['\psi'].append(self.psi)
        self.history['\delta'].append(delta)

        logging.info(f"Step completed. Position: [{round(self.position[0], 4)}, {round(self.position[1], 4)}], Reward: {round(reward, 4)}, Done: {done}, now_WP: [{round(self.target_position[0], 4)}, {round(self.target_position[1], 4)}]")

        return observation, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.copy(self.initial_state)
        self.position = np.array([0.0, 0.0])
        self.psi = 0.0
        self.history = {'x': [], 'y': [], 'rewards': [], '\psi': [], '\delta': []}
        self.current_waypoint_index = 0
        self.target_position = self.waypoints[self.current_waypoint_index]

        self.current_step = 0
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_error = np.arctan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])
        logging.info("Environment reset.")

        return np.array([self.position[0], self.position[1], self.state[0], self.state[1], self.state[2], distance_to_target, heading_error]), {}


    def render(self, render_mode='human', save_path='./plots'):
        if render_mode == 'human':
            logging.info("Rendering the environment.")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(5, 10))
            gs = gridspec.GridSpec(4, 2)
            
            # Plot trajectory and goal points
            ax1 = plt.subplot(gs[0:2, 0:2])
            ax1.plot(self.history['y'], self.history['x'], label='Trajectory')
            for wp in self.waypoints:
                ax1.scatter(wp[1], wp[0], color='red')  # 목표 위치 표시
                circle = plt.Circle((wp[1], wp[0]), 64, color='r', fill=False, linestyle='--')
                ax1.add_artist(circle)
            ax1.set_xlabel('Y Position')
            ax1.set_ylabel('X Position')
            ax1.set_xlim(-10, 300)
            ax1.set_ylim(-10, 300)
            ax1.set_title(f'Ship trajectory, now_wp : {self.target_position}')
            ax1.legend()

            # Plot rewards over time
            ax2 = plt.subplot(gs[2, 0:2])
            ax2.plot(self.history['rewards'], label='rewards', color = 'green')
            ax2.set_xlabel('step')
            ax2.set_ylabel('reward')
            ax2.set_title('rewards over time')
            ax2.legend()

            # Plot heading angle over time
            ax3 = plt.subplot(gs[3, 0:2])
            ax4 = ax3.twinx()
            ax3.plot(np.rad2deg(self.history['\psi']), label='psi', color='blue')
            ax3.plot(np.rad2deg(self.history['\delta']), label='delta', color='green')
            ax3.set_xlabel('step')
            ax3.set_ylabel('angle (deg)')
            ax4.set_ylabel('delta (deg)')
            ax3.set_title('psi and delta')
            ax3.legend()
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}/trajectory_and_rewards_env.png")
            plt.close()

            logging.info(f"Render completed and saved for environment.")

    # def render(self, render_mode='human', save_path='./plots', env_index=None):
            
            # # Plot trajectory and goal points
            # ax1 = plt.subplot(gs[0:2, 0:2])
            # ax1.plot(self.history['y'], self.history['x'], label='Trajectory')
            # ax1.scatter(self.target_position[1], [self.target_position[0]], color='red')  # 목표 위치 표시
            # circle = plt.Circle((self.target_position[1], self.target_position[0]), 64, color='r', fill=False, linestyle='--')
            # ax1.add_artist(circle)
            # ax1.set_xlabel('Y Position')
            # ax1.set_ylabel('X Position')
            # ax1.set_title(f'Ship Trajectory (Env {env_index}), wp : ({self.target_position})')
            # ax1.legend()

            # plt.tight_layout()
            # plt.savefig(f"{save_path}/trajectory_and_rewards_env{env_index}.png")
            # plt.close()

            # logging.info(f"Render completed and saved for environment {env_index}.")
            
    def multivariate_gaussian_reward(self, distance, heading_error):
        mean = np.array([0, 0])
        cov = np.array([[1, 0], [0, 1]])
        pos = np.array([distance, heading_error])
        rv = multivariate_normal(mean, cov)
        reward = rv.pdf(pos)
        return reward * 1000  # 스케일링을 통해 보상 값 조정
    
    def close(self):
        pass