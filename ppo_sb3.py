from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

from gym.wrappers import GrayScaleObservation

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from pathlib import Path
import random, datetime
import os

import time
from ppo_men import MemoryCallback
from memory_profiler import memory_usage
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    log_dir = './monitor_log/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env,4, channels_order='last')
    return env


def train(time_steps):
    env = make_env()

    learning_rate = 1e-6
    n_steps = 512
    log_dir = "./ppo_tensorboard/"
    model = PPO("CnnPolicy", env, verbose=1,tensorboard_log= log_dir, learning_rate=learning_rate,n_steps=n_steps)
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('PPO_%Y-%m-%dT%H-%M-%S')
    mem_callback = MemoryCallback()
    callback_func=CheckpointCallback(save_freq=200, save_path=save_dir.__str__())
    combined_callback = [callback_func, mem_callback]
    model.learn(total_timesteps=time_steps, callback=combined_callback)
    model.save(save_dir.__str__() + "ppo.model")


def test(path):
    env = make_env()
    model_dir = path
    #model_dir = "ppo.model"
    model = PPO.load(model_dir)
    obs = env.reset()
    obs = obs.copy()
    done = True
    while True:
        if done:
            state = env.reset()
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        #time.sleep(0.01)
        obs = obs.copy()
        env.render()


if __name__ == '__main__':
    #train(1000)
    test("./checkpoints/ppo_10k.model")