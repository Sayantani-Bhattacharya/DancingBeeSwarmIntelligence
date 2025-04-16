import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize
from Emulator import BeeSimEnv
import tqdm
import time
import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(arena_length, arena_width, num_bees, num_sources,robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity):
    def _init():
        env = BeeSimEnv(
            arena_length=arena_length,
            arena_width=arena_width,
            num_bees=num_bees,
            num_sources=num_sources,
            robot_distance_between_wheels=robot_distance_between_wheels,
            robot_wheel_radius=robot_wheel_radius,
            max_wheel_velocity=max_wheel_velocity,
            action_mode="multi"
            # "multi"
        )
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model for the herding task.")
    parser.add_argument("--num_bees", type=int, default=4, help="Number of bees in the simulation.")
    parser.add_argument("--num_sources", type=int, default=1, help="Number of sources in the simulation.")
    parser.add_argument("--model_path", type=str, default="trained_models/model_final.zip", help="Path to the trained RL model.")  #required = "True"
    parser.add_argument("--save_video", type=str, default="False", help="Save videos of simulations (True/False).")
    parser.add_argument("--num_sims", type=int, default=10, help="Number of simulations to run.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "offscreen"], help="Render mode for the environment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Convert save_video to boolean
    name = "Eval"
    time_now = time.strftime("%Y%m%d-%H%M%S")
    save_video = args.save_video.lower() == "true" 
    run = wandb.init(project='bee_swarm_rl', name=f"{name}-{time_now}" , sync_tensorboard=True, save_code=True)

    # create a video directory if it does not exist
    if save_video:
        import os
        if not os.path.exists("videos"):
            os.makedirs("videos")

    # Environment parameters
    arena_length = 20
    arena_width = 20
    robot_wheel_radius = 0.1
    robot_distance_between_wheels = 0.2
    max_wheel_velocity = 10.0

    env = make_env(
        arena_length, arena_width, args.num_bees, args.num_sources, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity
    )()

    # Load model
    models = {}
    for i in range(args.num_bees):
        models[i] = PPO.load(args.model_path, env=env, device='cpu')

    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successful_episodes': 0,
        'unsuccessful_episodes': 0,
        'success_rate': 0,
    }

    # Evaluate the model
    print(f"Starting evaluation of models")
    with torch.no_grad():
        for sim in tqdm.tqdm(range(args.num_sims)):
            observations = {}
            observations_array, _ = env.reset()
            i = 0
            for obs in observations_array:
                observations[i] = obs
                i +=1           

            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            # Per bee information.
            reward_dict, nectar_collect_reward_dict, nectar_delivery_reward_dict, dance_reward_dict, wiggle_obs_reward_dict = {}, {}, {}, {}, {}

            while not terminated and not truncated:
                episode_reward = 0

                for i in range(args.num_bees):
                    print("Length of observations: ", len(observations))
                    # print("Observations: ", observations)
                    
                    action, _ = models[i].predict(observations[i], deterministic=False)
                    observations[i], reward, nectar_collect_reward, nectar_delivery_reward, dance_reward, wiggle_obs_reward, terminated, truncated, _ = env.step(action, robot_id=i)
                    episode_reward += reward
                    reward_dict[i] = reward
                    nectar_collect_reward_dict[i] = nectar_collect_reward
                    nectar_delivery_reward_dict[i] = nectar_delivery_reward
                    dance_reward_dict[i] = dance_reward
                    wiggle_obs_reward_dict[i] = wiggle_obs_reward


                    
                episode_length += 1

                if args.render_mode == "human":
                    env.render(mode="human", fps=60)
                elif args.render_mode == "offscreen":
                    env.render()

                # Log metrics cummulatively for all bee.
                wandb.log({
                    # sum of elemetns of dict:
                    "reward/total": sum(reward_dict.values()),
                    "reward/nectar_collect": sum(nectar_collect_reward_dict.values()),
                    "reward/nectar_delivery": sum(nectar_delivery_reward_dict.values()),
                    "reward/dance": sum(dance_reward_dict.values()),
                    "reward/wiggle_obs": sum(wiggle_obs_reward_dict.values()),
                })

            # # Save video if specified
            # if save_video:
            #     print(f"Saving video for simulation {sim}")
            #     env.save_video(f"videos/simulation_{sim}.mp4", fps=60)
            # env.reset_frames()

            # Record metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            if terminated:
                metrics['successful_episodes'] += 1
            elif truncated:
                metrics['unsuccessful_episodes'] += 1

            

        # Close the environment
        env.close()

    # Calculate success rate
    metrics['success_rate'] = metrics['successful_episodes'] / args.num_sims

    # Display metrics
    print(f"Model evaluation complete")
    print(f"Average episode reward: {np.mean(metrics['episode_rewards'])}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths'])}")
    print(f"Average episode time: {np.mean(metrics['episode_lengths']) * 0.1} seconds")
    print(f"Success rate: {metrics['success_rate']}")
    print(f"Number of successful episodes: {metrics['successful_episodes']}")
    print(f"Number of unsuccessful episodes: {metrics['unsuccessful_episodes']}")
