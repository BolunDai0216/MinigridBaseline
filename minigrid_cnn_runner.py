import argparse
import time
from datetime import datetime

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from minigrid_cnn import MinigridCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-Empty-16x16-v0", help="MiniGrid env to use")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--iters", type=int, default=2e5, help="number of iterations to train")
    parser.add_argument("--save_freq", type=int, default=5e4, help="save model every n iterations")
    parser.add_argument("--seed", type=int, default=123, help="seed for PPO")
    parser.add_argument("--load_model", default=None, help="load a pretrained model")
    parser.add_argument("--render", action="store_true", help="render trained models")
    args = parser.parse_args()

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Create time stamp of experiment
    stamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    if args.train:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=f"./models/ppo/{args.env}_{stamp}/",
            name_prefix="iter",
        )

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"./logs/ppo/{args.env}_tensorboard/",
            seed=args.seed,
            device="cuda",
        )
        model.learn(
            args.iters,
            tb_log_name=f"{stamp}",
            callback=checkpoint_callback,
        )
    else:
        print(f"Testing model: models/ppo/{args.load_model}")

        ppo = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        ppo = ppo.load(f"models/ppo/{args.load_model}")

        obs, info = env.reset()
        rewards = 0

        for i in range(2000):
            action, _state = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward

            if terminated or truncated:
                print(f"Test reward: {rewards}")
                obs, info = env.reset()
                rewards = 0
                continue

        print(f"Test reward: {rewards}")

    env.close()


if __name__ == "__main__":
    main()
