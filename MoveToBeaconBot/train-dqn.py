import datetime
import time
import torch
from absl import app
from stable_baselines3 import DQN

from MoveToBeaconBot.MoveToBeaconAgentDiscrete import MoveToBeaconAgentDiscrete
import os

os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'


def main(unused_argv):
    device = torch.device('cuda')
    torch.set_default_device(device)
    print(device)

    policy = 'CnnPolicy'
    TIMESTEPS = 20000

    param_variations = [
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},

        # Learning rate variations
        {"learning_rate": 1e-5, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 5e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 1e-3, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},

        # Buffer size variations
        {"learning_rate": 1e-4, "buffer_size": 500000, "batch_size": 32, "gamma": 0.99, "target_update_interval": 10000,
         "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 50000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 250000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},

        # Batch size variations
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 16, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 64, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 128, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 4},

        # Gamma variations
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.95,
         "target_update_interval": 10000, "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.9, "target_update_interval": 10000,
         "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.85,
         "target_update_interval": 10000, "train_freq": 4},

        # Target update interval variations
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99, "target_update_interval": 5000,
         "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 20000, "train_freq": 4},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 50000, "train_freq": 4},

        # Train frequency variations
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 2},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 8},
        {"learning_rate": 1e-4, "buffer_size": 1000000, "batch_size": 32, "gamma": 0.99,
         "target_update_interval": 10000, "train_freq": 16},
    ]

    for i, params in enumerate(param_variations, start=1):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"MoveToBeaconAgent: {date}"

        models_dir = f"models/{model_name}/"
        logdir = f"logs/{model_name}/"

        env = MoveToBeaconAgentDiscrete()
        model = DQN(policy, env, verbose=1, tensorboard_log=logdir,
                    learning_rate=params["learning_rate"],
                    buffer_size=params["buffer_size"],
                    batch_size=params["batch_size"],
                    gamma=params["gamma"],
                    target_update_interval=params["target_update_interval"],
                    train_freq=params["train_freq"],
                    device='cuda')
        print(f"Training with parameters {params}")
        start_time = time.time()

        for episode in range(1, 101):
            print("On iteration: ", episode)
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
            model.save(f"{models_dir}/{TIMESTEPS * episode}")
            env.reset()

        end_time = time.time()
        duration = end_time - start_time

        with open(f"{models_dir}/args.txt", "w") as f:
            f.write(f"policy: {policy}\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")

        with open(f"{models_dir}/training_duration.txt", "w") as f:
            f.write(f"Training duration: {duration} seconds\n")

        env.close()

    print("Treinamento Finalizado")
    exit(0)


if __name__ == "__main__":
    app.run(main)
