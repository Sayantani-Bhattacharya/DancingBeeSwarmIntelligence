from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv, VecNormalize
import os
from Emulator import BeeSimEnv
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import imageio


STOPING_INTERATION = 2

# custom function to create environment instances
def make_env():
    def _init():
        # Simulation initialization data
        arena_length = 20  # meters
        arena_width = 20   # meters
        num_bees = 4
        num_sources = 1
        robot_wheel_radius = 0.1  # meters
        robot_distance_between_wheels = 0.2  # meters
        max_wheel_velocity = 8.0  # m/s

        # Create the environment
        env = BeeSimEnv(arena_length, arena_width, num_bees, num_sources, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, action_mode="multi")
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

    # === Frame collection for video ===
    video_frames = []

    ## Setting up the environment.
    num_envs = 1 # When actual training use 20  ||   # number of parallel environments
    
    
    # === Environment Parameters ===
    env = BeeSimEnv(
        arena_length=20,
        arena_width=20,
        num_bees=4,
        num_sources=1,
        robot_distance_between_wheels=0.2,
        robot_wheel_radius=0.1,
        max_wheel_velocity=8.0,
        action_mode="multi"
    )

    num_bees = env.num_bees
    TIMESTEPS = 5000
    EPISODES = 100
        
    # Vectorized env.
    # env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # # To run everything in a single thread: runs all environments in the main process.
    # # env = DummyVecEnv([make_env() for _ in range(num_envs)])
    # env = VecMonitor(env)  # VecMonitor wraps the entire VecEnv for logging
    # env = VecNormalize(env, norm_reward=True) # VecNormalize normalizes the rewards

    # Initialize the model
    model = PPO('MlpPolicy', env, verbose=1, device="cpu", n_steps=6144, tensorboard_log=logdir)
    TIMESTEPS = 1 #250000 # number of timesteps to train the model for before logging
    # calculate iterations based on num_timesteps
    iters = model.num_timesteps // TIMESTEPS
    print(f"Starting from iteration {iters}")

    observations = {}
    observations_array, _ = env.reset()

    # Main training loop.

    # This is for swarm loop.
    for ep in range(EPISODES):
        print(f"\n=== EPISODE {ep + 1} ===")

        i = 0
        for obs in observations_array:
            observations[i] = obs
            i +=1

        total_reward = 0

        for step in range(TIMESTEPS):
            for i in range(num_bees):
                obs = observations[i]
                action, _ = model.predict(obs, deterministic=False)

                # Step for only one robot
                new_obs, reward, terminated, truncated, _ = env.step(action, robot_id=i)
                observations[i] = new_obs
                total_reward += reward

            if step % 20 == 0:
                env.render(mode="human", fps=60)
                frame = env.render(mode="rgb_array")
        
        video_frames = env.get_video_frames()
        env.reset_frames()

        print(f"Episode {ep + 1} finished with total reward: {total_reward}")

        # # === Save model periodically ===
        # if (ep + 1) % 10 == 0:
        #     model.save(f"{models_dir}/bee_model_ep{ep + 1}")

        # log the video
        # === Save and log video to W&B ===
        video_path = "videos/bee_eval_run.mp4"
        
        wandb.log({
            "episode_reward": total_reward,
            "episode_length": step,
            "video": wandb.Video(video_frames, caption="Eval run", format="mp4", fps=30)
        })

        # Save the final model as a zip file
        model.save(f"{models_dir}/bee_model_final.zip")
        model.save(f"trained_models/model_final.zip")


        print("🎥 Video logged to wandb!")

        # === Final cleanup ===
        # env.reset()

    # This is the one for single agent training.
    # # while True:
    # if (iters <= STOPING_INTERATION):
    #     iters += 1
    #     print(f"Starting iteration {iters}...")
    #     # custom_step = TIMESTEPS*iters

    #     # model.learn(total_timesteps= TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(model_save_freq=TIMESTEPS, model_save_path=f"{models_dir}/{TIMESTEPS*iters}", verbose=1))
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(model_save_freq=TIMESTEPS, model_save_path=f"{models_dir}/{TIMESTEPS*iters}", verbose=1))

    #     print(f"Completed iteration {iters}.")

    #     # render a video of the trained model in action
    #     single_env = DummyVecEnv([make_env()])
    #     obs,_ = single_env.reset()
    #     done = False
    #     while not done:
    #         print("Rendering video...")
    #         action, _states = model.predict(obs)
    #         obs, reward, done, info = single_env.step(action)
    #         single_env.envs[0].render(mode="human", fps=60)

    #     # log the video
    #     video_frames = single_env.envs[0].get_video_frames()
    #     single_env.envs[0].reset_frames()
    #     wandb.log({"video": wandb.Video(video_frames, caption=f"Model at iteration {iters}",format="mp4", fps=30)})
    #     print(f"Video logged at iteration {TIMESTEPS*iters}")
    #     single_env.reset()