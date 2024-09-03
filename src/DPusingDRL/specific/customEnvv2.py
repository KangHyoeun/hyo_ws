#!/usr/bin/env python3
# customEnv.py

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
from utils.tools import ssa
import math
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VesselEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vessel: Vessel, initial_state: np.ndarray, dT: float, target_position: np.ndarray, render_mode='human', max_steps=1000):
        super(VesselEnv, self).__init__()
        self.vessel = vessel
        self.render_mode = render_mode
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.dT = dT
        self.target_position = target_position

        self.psi = 0.0  # Assume initial heading is zero
        self.position = np.array([0.0, 0.0])  # Assume initial position is at the origin

        # 환경외란 설정
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

        # Observation space: [x, y, u, v, r, distance to target, heading error]
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
        # ssa 적용
        self.psi = ssa(self.psi)

        # Update position
        self.position[0] += self.state[0] * np.cos(self.psi) * self.dT - self.state[1] * np.sin(self.psi) * self.dT
        self.position[1] += self.state[0] * np.sin(self.psi) * self.dT + self.state[1] * np.cos(self.psi) * self.dT

        # Calculate distance to target and heading error
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        heading_error = np.arctan2(self.target_position[1] - self.position[1],
                                   self.target_position[0] - self.position[0]) - self.psi
        
        heading_error = ssa(heading_error)
        

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

        # 3. 목표 위치와의 거리에 따른 보상
        reward -= distance_to_target*0.1

        # 2. 방위각 오차에 따른 보상
        reward -= abs(heading_error)

        # 5. 안정적인 경로 유지 보상 (조작량 줄이기)
        # reward -= abs(delta) * 10.0
        
        # 다변량 가우시안 보상 함수
        # reward = self.multivariate_gaussian_reward(distance_to_target, heading_error)
        
        # Done condition
        terminated = distance_to_target < 32.0  # 1/2L 정도 가까워지고 동시에 heading_error도 5도 이내일때
        truncated = self.current_step >= self.max_steps

        done = terminated or truncated  # Combine conditions for done

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
            ax1.plot(self.history['y'], self.history['x'], label='trajectory', color='green')
            ax1.scatter(self.target_position[1], [self.target_position[0]], color='red')  # 목표 위치 표시
            circle = plt.Circle((self.target_position[1], self.target_position[0]), 32, color='r', fill=False, linestyle='--')
            ax1.add_artist(circle)
            ax1.set_xlabel('Y Position')
            ax1.set_ylabel('X Position')
            ax1.set_title(f'Ship trajectory, wp : ({self.target_position})')
            ax1.legend()

            # Plot rewards over time
            ax2 = plt.subplot(gs[2, 0:2])
            ax2.plot(self.history['rewards'], label='rewards', color = 'cyan')
            ax2.set_xlabel('step')
            ax2.set_ylabel('reward')
            ax2.set_title('rewards over time')
            ax2.legend()

            # Plot heading angle over time
            ax3 = plt.subplot(gs[3, 0:2])
            ax4 = ax3.twinx()
            ax3.plot(np.rad2deg(self.history['\psi']), label='psi', color='blue')
            ax3.set_xlabel('step')
            ax3.set_ylabel('angle (deg)')
            ax3.set_title('psi over time')
            ax3.legend()
            ax4.legend()

            plt.tight_layout()
            plt.savefig(f"{save_path}/trajectory_and_rewards_env.png")
            plt.close()

            logging.info(f"Render completed and saved for environment.")
                
    def multivariate_gaussian_reward(self, distance, heading_error):
        mean = np.array([0, 0])
        cov = np.array([[1, 0], [0, 1]])
        pos = np.array([distance, heading_error])
        rv = multivariate_normal(mean, cov)
        reward = rv.pdf(pos)
        return reward * 1000  # 스케일링을 통해 보상 값 조정
    
    def close(self):
        pass
