import datetime
import time
import torch
from absl import app
from stable_baselines3 import PPO

from MoveToBeaconBot.MoveToBeaconAgent import MoveToBeaconAgent


def main(unused_argv):
    model_file = 'xxxx.zip'
    model_name = "MoveToBeaconAgent: 2024-09-07_18-33-48"
    model_path = f"models/{model_name}/{model_file}"

    logdir = f"logs/{model_name}/"

    device = torch.device('cuda')
    torch.set_default_device(device)
    print(device)

    env = MoveToBeaconAgent()

    model = PPO.load(f"{model_path}", env=env, device=device)

    num_episodes = 1
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episódio {episode} finalizado com recompensa total: {total_reward}")

    env.close()
    print("Teste Finalizado")
    exit(0)


if __name__ == "__main__":
    app.run(main)
