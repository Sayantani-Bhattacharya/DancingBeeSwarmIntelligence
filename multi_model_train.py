from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv, VecNormalize
import os
from Emulator import BeeSimEnv
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np


STOPING_INTERATION = 2

# custom function to create environment instances
def make_env():
    def _init():
        # Simulation initialization data
        arena_length = 15  # meters
        arena_width = 15   # meters
        num_bees = 8
        num_sources = 1
        # num_sheep = 1
        # num_sheepdogs = 1
        robot_wheel_radius = 0.1  # meters
        robot_distance_between_wheels = 0.2  # meters
        max_wheel_velocity = 8.0  # m/s

        # Create the environment
        env = BeeSimEnv(arena_length, arena_width, num_bees, num_sources, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, action_mode="point")

        # print(f"Environment initialized in process ID: {os.getpid()}")

        return env
    return _init

if __name__ == "__main__":
    # Initialize wandb for logging
    name = "bee_swarm_PPO"
    time_now = time.strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project='bee_swarm_rl', name=f"{name}-{time_now}" , sync_tensorboard=True, save_code=True)

    # Create directories for saving models and logs
    models_dir = f"models/bee_swarm_test/{name}-{time_now}"
    logdir = f"logs/bee_swarm_test/{name}-{time_now}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    num_envs = 20 # number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)  # VecMonitor wraps the entire VecEnv for logging
    env = VecNormalize(env, norm_reward=True) # VecNormalize normalizes the rewards

    # Initialize the model
    model = PPO('MlpPolicy', env, verbose=1, device="cpu", n_steps=6144, tensorboard_log=logdir)
    TIMESTEPS = 1 #250000 # number of timesteps to train the model for before logging
    # calculate iterations based on num_timesteps
    iters = model.num_timesteps // TIMESTEPS
    print(f"Starting from iteration {iters}")

    # Main training loop
    # while True:
    if (iters <= STOPING_INTERATION):
        iters += 1
        print(f"Starting iteration {iters}...")
        # custom_step = TIMESTEPS*iters

        # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(model_save_freq=TIMESTEPS, model_save_path=f"{models_dir}/{TIMESTEPS*iters}", verbose=1))
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(model_save_freq=TIMESTEPS, model_save_path=f"{models_dir}/{TIMESTEPS*iters}", verbose=1))

        
        print(f"Completed iteration {iters}.")

        # render a video of the trained model in action
        single_env = DummyVecEnv([make_env()])
        obs = single_env.reset()
        done = False
        while not done:
            print("Rendering video...")
            action, _states = model.predict(obs)
            obs, reward, done, info = single_env.step(action)
            single_env.envs[0].render(mode="human", fps=60)

        # log the video
        video_frames = single_env.envs[0].get_video_frames()
        single_env.envs[0].reset_frames()
        wandb.log({"video": wandb.Video(video_frames, caption=f"Model at iteration {iters}",format="mp4", fps=30)})
        print(f"Video logged at iteration {TIMESTEPS*iters}")
        single_env.reset()