#!/usr/bin/env python3
import torch
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from IPython import display
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
import math
import stable_baselines3
from stable_baselines3 import A2C

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt=1.0):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class ShipDockingEnv(gym.Env):
    def __init__(self):
        super(ShipDockingEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)  # PID의 조정값
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)  # x, y, theta, error

        self.goal = np.array([100, 0])  # 목표 위치
        self.pid = PIDController(kp=0.3, ki=0.01, kd=0.05)
        self.state = np.array([0, 0, 0, np.linalg.norm([100, 0])])
        self.history = {'x': [], 'y': [], 'rewards': []}

    def reset(self):
        self.state = np.array([0, 0, 0, np.linalg.norm([100, 0])])  # 초기 위치 및 오차 x, y, theta, error
        self.history = {'x': [self.state[0]], 'y': [self.state[1]], 'rewards': []}
        return self.state

    def step(self, action):
        x, y, theta, error = self.state
        rudder_angle = self.pid.update(self.state[3]) + action[0] * 0.05  # PID 조정 및 액션 적용
        self.state[2] += rudder_angle  # 방향 갱신
        self.state[0] += np.cos(self.state[2])  # x 위치 갱신
        self.state[1] += np.sin(self.state[2])  # y 위치 갱신
        self.state[3] = np.linalg.norm(self.goal - self.state[:2])  # 오차 갱신

        done = self.state[3] < 1  # 종료 조건
        reward = -self.state[3]  # 보상은 목표에 가까워질수록 증가

        self.history['x'].append(self.state[0])
        self.history['y'].append(self.state[1])
        self.history['rewards'].append(reward)

        return self.state, reward, done, {}

    def render(self, mode='human', save_path='./plots'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['x'], self.history['y'], label='Trajectory')
        plt.scatter([self.goal[0]], [self.goal[1]], color='red', label='Goal')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Ship Trajectory')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['rewards'], label='Rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Rewards Over Time')
        plt.legend()

        plt.savefig(f"{save_path}/trajectory_and_rewards.png")
        plt.close()

    def close(self):
        pass

# 환경 및 모델 초기화
env = ShipDockingEnv()
model = A2C("MlpPolicy", env, verbose=1)

# 학습 시작 시간 기록
start_time = time.time()

# 모델 학습
obs = env.reset()
total_reward = 0
for e in range(15000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"episode: {e+1}, rewards: {rewards}")
    env.render()
    if dones:
        obs = env.reset()

# 학습 종료 시간 기록 및 출력
end_time = time.time()
elapsed_time = end_time - start_time
print(f"학습 완료! 총 걸린 시간: {elapsed_time//60:.0f} 분 {elapsed_time%60:.0f} 초")

# 학습된 모델 저장
model.save("a2c_ship_docking")

# 환경 닫기
env.close()
