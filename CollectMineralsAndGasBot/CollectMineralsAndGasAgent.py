import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_SELF = features.PlayerRelative.SELF

_UNIT_TYPE = 6
_SELECTED = 7
_UNIT_HIT_POINTS = 8
_RESOURCES = [units.Neutral.MineralField, units.Neutral.VespeneGeyser]


class CollectMineralsAndGasAgent(gym.Env):
    def __init__(self):
        super(CollectMineralsAndGasAgent, self).__init__()
        self.current_timestep = None
        self.cumulative_minerals = 0
        self.cumulative_gas = 0
        self.refinery_count = 0
        self.features = [_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS]
        self.env = sc2_env.SC2Env(
            map_name="CollectMineralsAndGas",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=48, minimap=48),
                use_feature_units=True
            ),
            step_mul=None,
            game_steps_per_episode=None,
            visualize=False,
            replay_dir='./replays',
            save_replay_episodes=0,
            realtime=False
        )

        self.action_space = spaces.MultiDiscrete([8, 4, 48, 48])
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 48, 48), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        timesteps = self.env.reset()
        initial_obs = self._extract_observation(timesteps[0])
        self.current_timestep = timesteps[0]
        self.cumulative_minerals = 0
        self.cumulative_gas = 0
        self.refinery_count = 0
        return initial_obs, {}

    def step(self, action):
        sc2_action = self._transform_action(action, self.current_timestep)
        timesteps = self.env.step([sc2_action])
        self.current_timestep = timesteps[0]
        obs = self._extract_observation(timesteps[0])

        reward = self._calculate_reward(timesteps[0])
        done = timesteps[0].last()
        truncated = False
        return obs, reward, done, truncated, {}

    def _calculate_reward(self, timestep):
        player = timestep.observation['player']
        current_minerals = player[1]
        current_gas = player[2]

        game_units = timestep.observation['feature_units']

        resource_reward = (current_minerals - self.cumulative_minerals) + (current_gas - self.cumulative_gas)

        units_timestamp = timestep.observation['feature_units']
        refineries = [unit for unit in game_units if unit.unit_type == units.Terran.Refinery]
        new_refineries = len(refineries) - self.refinery_count

        construction_reward = new_refineries * 10

        self.cumulative_minerals = current_minerals
        self.cumulative_gas = current_gas
        self.refinery_count = len(refineries)

        return resource_reward + construction_reward

    def _extract_observation(self, timestep):
        screen = np.array(timestep.observation['feature_screen'][self.features], dtype=np.uint8)
        return screen

    def _select_scv(self, timestep, scv_index):
        game_units = timestep.observation['feature_units']
        scvs = [unit for unit in game_units if unit.unit_type == units.Terran.SCV]
        if len(scvs) > scv_index:
            scv = scvs[scv_index]
            if 0 <= scv.y < 48 and 0 <= scv.x < 48:
                return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
            else:
                return actions.FUNCTIONS.no_op()
        return actions.FUNCTIONS.no_op()

    def _transform_action(self, action, timestep):
        if timestep is None:
            raise ValueError("Timestep is not initialized.")

        scv_index, action_type, target_x, target_y = action
        available_actions = timestep.observation.available_actions

        if action_type == 0:
            select_action = self._select_scv(timestep, scv_index)
            return select_action

        if action_type == 1:  # Mover
            if actions.FUNCTIONS.Move_screen.id in available_actions:
                return actions.FUNCTIONS.Move_screen("now", (target_x, target_y))
            else:
                return actions.FUNCTIONS.no_op()

        elif action_type == 2:  # Construir Refinaria
            if actions.FUNCTIONS.Build_Refinery_screen.id in available_actions:
                return actions.FUNCTIONS.Build_Refinery_screen("now", (target_x, target_y))
            else:
                return actions.FUNCTIONS.no_op()

        elif action_type == 3:  # Coletar (Harvest_Gather_screen)
            if actions.FUNCTIONS.Harvest_Gather_screen.id in available_actions:
                return actions.FUNCTIONS.Harvest_Gather_screen("now", (target_x, target_y))
            else:
                return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.no_op()

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()
