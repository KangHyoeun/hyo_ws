#!/usr/bin/env python3

# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Image, Figure, Video, HParam, TensorBoardOutputFormat

import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel, InitialValues
import os

from typing import Any, Dict
import gymnasium as gym
import torch as th
import time
import numpy as np

from customEnvv2 import VesselEnv
from wptEnv import ShipEnv

@dataclass
class KVLCC2Inits:
    full_scale = InitialValues(
        u     = 3.85, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 1.05 # Propeller revs [s⁻¹]
    )
    
    l_64 = InitialValues(
        u     = 4.0, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 3.0 # Propeller revs [s⁻¹]
    )
    
    l_7 = InitialValues(
        u     = 1.128, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 13.4 # Propeller revs [s⁻¹]
    )


# Use a pre-calibrated vessel
vessel = Vessel(**cvs.kvlcc2_l64)
ivs = KVLCC2Inits.l_64

# 초기 상태 정의
initial_state = np.array([ivs.u, ivs.v, ivs.r], dtype=np.float64)
print("Initial state set.")

# 경유점 정의
# x_coordinate = np.random.randint(50, 501)  # x 좌표는 50 ~ 500 사이의 정수
# y_coordinate = np.random.randint(-500, 501)  # y 좌표는 -500 ~ 500 사이의 정수
target_position = np.array([100.0, 100.0], dtype=np.float64)
print(f"Target position set: {target_position}")

# waypoints = [
#     np.array([250, 250]),
#     np.array([500, 250]),
#     np.array([500, -250]),
#     np.array([250, -250])
# ]
# print("Waypoints set.")

env = VesselEnv(vessel=vessel, initial_state=initial_state, dT=0.1, target_position=target_position)

# 모델 훈련 및 저장 디렉토리 설정
log_dir = "./ppo_dp_train_model/"
os.makedirs(log_dir, exist_ok=True)

# 모델 저장 경로
model_save_path = os.path.join(log_dir, "ppo_dp")

# 학습 시간 측정을 위한 시작 시간 기록
start_time = time.time()

# PPO 모델 초기화
model = PPO("MlpPolicy", env, verbose=1)

# 모델 훈련 (예: 10000 타임스텝)
model.learn(total_timesteps=10000)

# 학습 완료 후 시간 측정
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# 모델 저장
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 학습된 모델 로드
model = PPO.load(model_save_path)

# 모델 평가: 10번의 에피소드에서 평균 성능 평가
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# 모델 학습된 후 모델 테스트
obs, _ = env.reset()  # obs와 info 분리
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()  # obs와 info 분리
