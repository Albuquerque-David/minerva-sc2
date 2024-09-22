import datetime
import time
import torch
from absl import app
from stable_baselines3 import PPO, A2C

from FindAndDefeatZerglingsBot.FindAndDefeatZerglingsAgent import FindAndDefeatZerglingsAgent

def main(unused_argv):

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"FindAndDefeatZerglingsAgent: {date}"

    models_dir = f"models/{model_name}/"
    logdir = f"logs/{model_name}/"

    first_open = True

    device = torch.device('cuda')
    torch.set_default_device(device)
    print(device)

    policy = 'CnnPolicy'

    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    ent_coef = 0.01

    env = FindAndDefeatZerglingsAgent()
    model = PPO(policy,
                env, verbose=1, tensorboard_log=logdir,
                learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch_size, n_epochs=n_epochs,
                gamma=gamma, ent_coef=ent_coef
                )

    TIMESTEPS = 20000

    start_time = time.time()

    for i in range(1, 50):
        print("On iteration: ", i)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS * i}")
        if first_open:
            # Salvar os parametros do treinamento em um arquivo.txt
            with open(f"{models_dir}/args.txt", "w") as f:
                f.write(f"policy: {policy}\n")
                f.write(f"learning_rate: {learning_rate}\n")
                f.write(f"n_steps: {n_steps}\n")
                f.write(f"batch_size: {batch_size}\n")
                f.write(f"n_epochs: {n_epochs}\n")
                f.write(f"gamma: {gamma}\n")
                f.write(f"ent_coef: {ent_coef}\n")

            first_open = False
        env.reset()

    end_time = time.time()
    duration = end_time - start_time

    with open(f"{models_dir}/training_duration.txt", "w") as f:
        f.write(f"Training duration: {duration} seconds\n")

    env.close()
    print("Treinamento Finalizado")
    exit(0)


if __name__ == "__main__":
    app.run(main)
