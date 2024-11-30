import datetime
import time
import torch
from absl import app
from stable_baselines3 import PPO

from CollectMineralsShardsBot.CollectMineralsShardsAgent import CollectMineralsShardsAgent
import os

os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'


def main(unused_argv):
    device = torch.device('cpu')
    torch.set_default_device(device)
    print(device)

    policy = 'CnnPolicy'
    TIMESTEPS = 20000

    param_variations = [
        {'learning_rate': 0.0001, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},

        {'learning_rate': 0.001, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0001, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.00005, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},

        {'learning_rate': 0.0003, 'n_steps': 1024, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 4096, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 3072, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},

        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 32, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.0},

        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 5, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 15, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 20, 'gamma': 0.99, 'ent_coef': 0.0},

        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.95, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.999, 'ent_coef': 0.0},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.90, 'ent_coef': 0.0},

        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.05},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.1},
        {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'gamma': 0.99, 'ent_coef': 0.001}
    ]

    for i, params in enumerate(param_variations, start=1):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"CollectMineralsShardsAgent: {date}"

        models_dir = f"models/{model_name}/"
        logdir = f"logs/{model_name}/"

        env = CollectMineralsShardsAgent()
        model = PPO(policy, env, verbose=1, tensorboard_log=logdir,
                    learning_rate=params['learning_rate'],
                    n_steps=params['n_steps'],
                    batch_size=params['batch_size'],
                    n_epochs=params['n_epochs'],
                    gamma=params['gamma'],
                    ent_coef=params['ent_coef'],
                    device='cpu')
        print(f"Training with parameters {params}")
        start_time = time.time()

        for episode in range(1, 101):
            print("On iteration: ", episode)
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
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
