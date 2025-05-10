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
    name = "Train"
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

    # Setting up the environment.
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
    model = PPO('MlpPolicy', env, verbose=1, device="cuda", n_steps=6144, tensorboard_log=logdir)
    TIMESTEPS = 500 #250000 # number of timesteps to train the model for before logging [this is also the steps for each episode]
    # calculate iterations based on num_timesteps
    iters = model.num_timesteps // TIMESTEPS
    print(f"Starting from iteration {iters}")

    observations = {}
    # per_bee_reward: Dictionary that has total reward in indivudual step for each bee, with key as teh bee id.
    per_bee_reward = {}
    total_energy = 0
    per_bee_energy = {}
    observations_array, _ = env.reset()

    # Initialize per bee energy metrics
    nectar_collect_total = {i: 0 for i in range(num_bees)}
    nectar_delivery_total = {i: 0 for i in range(num_bees)}
    dance_total = {i: 0 for i in range(num_bees)}
    wiggle_obs_total = {i: 0 for i in range(num_bees)}


    # Main training loop.

    # This is for swarm loop.
    for ep in range(EPISODES):
        print(f"\n=== EPISODE {ep + 1} ===")

        i = 0
        for obs in observations_array:
            observations[i] = obs
            i +=1


        for step in range(TIMESTEPS):
            total_reward = 0


            for i in range(num_bees):
                obs = observations[i]
                action, _ = model.predict(obs, deterministic=False)
                # Step for only one robot
                new_obs, reward, nectar_collect_reward, nectar_delivery_reward, dance_reward, wiggle_obs_reward, terminated, truncated, _ = env.step(action, robot_id=i)
                observations[i] = new_obs
                
                # Storing the reward metrics for each bee.
                nectar_collect_total[i] += nectar_collect_reward
                nectar_delivery_total[i] += nectar_delivery_reward
                dance_total[i] += dance_reward
                wiggle_obs_total[i] += wiggle_obs_reward
                
                # Accumulating reward for each bee for logging.
                if i not in per_bee_reward:
                    per_bee_reward[i] = 0
                per_bee_reward[i] += reward 
                total_energy = env.robots[i].energy_level
                # Accumulating energy for each bee for logging.
                if i not in per_bee_energy:
                    per_bee_energy[i] = 0
                per_bee_energy[i] += env.robots[i].energy_level  

            if step % 20 == 0:
                env.render(mode="human", fps=60)

            wandb.log({
                "episode_length": step,
                "episode": ep,
                "reward/total_episode": sum(per_bee_reward.values()),
                "reward/bee0": per_bee_reward[0],
                "reward/bee1": per_bee_reward[1],
                "reward/bee2": per_bee_reward[2],
                "reward/bee3": per_bee_reward[3],

                "reward/nectar_collect_total": sum(nectar_collect_total.values()),
                "reward/nectar_delivery_total": sum(nectar_delivery_total.values()),
                "reward/dance_total": sum(dance_total.values()),
                "reward/wiggle_obs_total": sum(wiggle_obs_total.values()),

                "energy/bee0": per_bee_energy[0],
                "energy/bee1": per_bee_energy[1],
                "energy/bee2": per_bee_energy[2],
                "energy/bee3": per_bee_energy[3],

                # "video": wandb.Video(video_frames, caption="Eval run", format="mp4", fps=30)
            })

            # Reset the per bee energy metrics for each episode: for logging i dont need accumulated metrics, i need accumulation for enery and reward only.
            nectar_collect_total = {i: 0 for i in range(num_bees)}
            nectar_delivery_total = {i: 0 for i in range(num_bees)}
            dance_total = {i: 0 for i in range(num_bees)}
            wiggle_obs_total = {i: 0 for i in range(num_bees)}
            
        
        video_frames = env.get_video_frames()
        # print(f"Episode {ep + 1} finished with total reward: {total_reward}")

        # # === Save model periodically ===
        # if (ep + 1) % 10 == 0:
        #     model.save(f"{models_dir}/bee_model_ep{ep + 1}")

        # # TODO: Working!! just add this as optional, as rendering becomes really slow.  
        # # === Save and log video to W&B ===
        # video_path = "videos/bee_eval_run.mp4"
        # env.save_video(video_path, fps=60)
        # wandb.log({
        #     "video": wandb.Video(video_path, caption="Eval run", format="mp4", fps=60)
        # })
        # print("🎥 Video logged to wandb!")


        # Save the final model as a zip file
        model.save(f"{models_dir}/bee_model_final.zip")
        model.save(f"trained_models/model_final.zip")



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