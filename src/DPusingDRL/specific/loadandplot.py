#!/usr/bin/env python3

# loadandplot.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from wptEnv import ShipEnv
from customEnv import VesselEnv

# 초기 설정
import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass
from mmgdynamics.structs import Vessel, InitialValues

@dataclass
class KVLCC2Inits:
    full_scale = InitialValues(
        u     = 3.85,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 1.05
    )
    
    l_64 = InitialValues(
        u     = 4.0,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 3.0
    )
    
    l_7 = InitialValues(
        u     = 1.128,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 13.4
    )

# 미리 보정된 선박 사용
vessel = Vessel(**cvs.kvlcc2_l64)
ivs = KVLCC2Inits.l_64

# 초기 상태 정의
initial_state = np.array([ivs.u, ivs.v, ivs.r])
print("Initial state set.")

# 경유점 정의
# target_position = np.array([250, 250])
# print(f"Target position set: {target_position}")

waypoints = [
    np.array([250, 250])
]
print("Waypoints set.")

# 환경 생성 1

# env = VesselEnv(vessel, initial_state, dT=0.1, target_position=target_position, render_mode='human', max_steps=2000)
env = ShipEnv(vessel, initial_state, dT=0.1, waypoints=waypoints, render_mode='human', max_steps=2000)
env = Monitor(env, f"./logs/")

# PPO 모델 생성
model = PPO("MlpPolicy", env, verbose=1)
print("Model created.")

# 모델 로드
model = PPO.load("logs/real/best_model")
print("Best Model loaded.")

obs, info = env.reset()
done = False

max_timesteps = 10000  # 최대 타임스텝 수
current_timesteps = 0

while not done and current_timesteps < max_timesteps:
    action, _states = model.predict(obs)
    obs, rews, terms, truns, info = env.step(action)
    done = terms or truns
    current_timesteps += 1
    env.render()

if current_timesteps >= max_timesteps:
    print("Reached maximum timesteps.")
else:
    print("All environments completed.")