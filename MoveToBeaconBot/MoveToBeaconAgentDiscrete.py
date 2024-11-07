import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

_UNIT_TYPE = 6
_SELECTED = 7
_UNIT_HIT_POINTS = 8


class MoveToBeaconAgentDiscrete(gym.Env):
    def __init__(self):
        super(MoveToBeaconAgentDiscrete, self).__init__()
        self.current_timestep = None
        self.cumulative_reward = 0
        self.features = [_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS]
        self.env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=48, minimap=48),
                use_feature_units=True
            ),
            step_mul=None,
            game_steps_per_episode=None,
            visualize=False,
            replay_dir='./replays',
            save_replay_episodes=0
        )

        self.action_space = spaces.Discrete(48 * 48)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 48, 48), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        timesteps = self.env.reset()
        initial_obs = self._extract_observation(timesteps[0])
        self.current_timestep = timesteps[0]
        self.action_space = spaces.Discrete(48 * 48)
        self.cumulative_reward = 0
        info = {}
        return initial_obs, info

    def step(self, action):
        x, y = self._discrete_to_multi_discrete(action)

        sc2_action = self._transform_action(x, y, self.current_timestep)
        timesteps = self.env.step([sc2_action])
        self.current_timestep = timesteps[0]
        obs = self._extract_observation(timesteps[0])

        # Função de Recompensa
        reward = self._calculate_reward(timesteps[0])

        done = timesteps[0].last()
        info = {}
        truncated = False

        return obs, reward, done, truncated, info

    def _discrete_to_multi_discrete(self, action):
        """Converte a ação discreta em um par (x, y)."""
        x = action // 48
        y = action % 48
        return x, y

    def _calculate_reward(self, timestep):
        reward = 0
        timestep_cumulative_score = timestep.observation['score_cumulative'].score
        if timestep_cumulative_score > self.cumulative_reward:
            self.cumulative_reward += 1
            reward = self.cumulative_reward
            print(f'Recompensa recebida: {reward}')
        return reward

    def _extract_observation(self, timestep):
        screen = np.array(timestep.observation['feature_screen'][self.features], dtype=np.uint8)
        return screen

    def _select_unit_marine(self, timestep):
        game_units = timestep.observation['feature_units']
        marines = [unit for unit in game_units if unit.unit_type == units.Terran.Marine]
        return actions.FUNCTIONS.select_point("select", (marines[0].x, marines[0].y))

    def _transform_action(self, target_x, target_y, timestep):
        if timestep is None:
            raise ValueError("Timestep is not initialized.")

        if actions.FUNCTIONS.Move_screen.id in timestep.observation.available_actions:
            return actions.FUNCTIONS.Move_screen("now", (target_x, target_y))
        else:
            return self._select_unit_marine(timestep)

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()