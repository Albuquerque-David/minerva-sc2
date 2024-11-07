import datetime
import time
import torch
from absl import app
from stable_baselines3 import A2C

from MoveToBeaconBot.MoveToBeaconAgent import MoveToBeaconAgent

def main(unused_argv):


    device = torch.device('cpu')
    torch.set_default_device(device)
    print(device)

    policy = 'CnnPolicy'
    TIMESTEPS = 20000

    param_variations = [
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},

        {"learning_rate": 0.007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0003, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.00007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},

        {"learning_rate": 0.0007, "n_steps": 10, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 15, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 20, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},

        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.95, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.97, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.5},

        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.25, "ent_coef": 0.0, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.75, "ent_coef": 0.0, "max_grad_norm": 0.5},

        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5},
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.05, "max_grad_norm": 0.5},

        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.25},
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.0, "max_grad_norm": 0.75},

        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5},
    ]

    for i, params in enumerate(param_variations, start=1):
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"MoveToBeaconAgent: {date}"

        models_dir = f"models/{model_name}/"
        logdir = f"logs/{model_name}/"

        env = MoveToBeaconAgent()
        model = A2C(policy, env, verbose=1, tensorboard_log=logdir,
                    learning_rate=params["learning_rate"],
                    n_steps=params["n_steps"],
                    gamma=params["gamma"],
                    vf_coef=params["vf_coef"],
                    ent_coef=params["ent_coef"],
                    max_grad_norm=params["max_grad_norm"],
                    device='cpu')
        print(f"Training with parameters {params}")
        start_time = time.time()

        for i in range(1, 101):
            print("On iteration: ", i)
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
            model.save(f"{models_dir}/{TIMESTEPS * i}")
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
