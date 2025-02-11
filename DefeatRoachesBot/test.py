import datetime
import time
import torch
from absl import app
from stable_baselines3 import PPO

from DefeatRoachesBot.DefeatRoachesAgent import DefeatRoachesAgent


def main(unused_argv):
    model_file = 'xxxx.zip'
    model_name = "DefeatRoachesAgent: 2024-12-18_09-13-41"
    model_path = f"models/{model_name}/{model_file}"

    logdir = f"logs/{model_name}/"

    device = torch.device('cpu')
    torch.set_default_device(device)
    print(device)

    env = DefeatRoachesAgent()

    model = PPO.load(f"{model_path}", env=env, device=device)

    num_episodes = 100
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
