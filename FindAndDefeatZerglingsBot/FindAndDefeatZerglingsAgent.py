import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_SELF = features.PlayerRelative.SELF

_UNIT_TYPE = 6
_SELECTED = 7
_UNIT_HIT_POINTS = 8


class FindAndDefeatZerglingsAgent(gym.Env):
    def __init__(self):
        super(FindAndDefeatZerglingsAgent, self).__init__()
        self.current_timestep = None
        self.cumulative_reward = 0
        self.total_allied_units = 3
        self.cumulative_kills = 0
        self.features = [_PLAYER_RELATIVE, _UNIT_TYPE, _UNIT_HIT_POINTS]
        self.env = sc2_env.SC2Env(
            map_name="FindAndDefeatZerglings",
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

        # Action space: [Select Mask (3 bits), Action Type (Move/Attack), X, Y]
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 4, 48, 48])
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 48, 48), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        timesteps = self.env.reset()
        initial_obs = self._extract_observation(timesteps[0])
        self.current_timestep = timesteps[0]
        self.cumulative_reward = 0
        self.cumulative_kills = 0
        self.total_allied_units = len([
            unit for unit in timesteps[0].observation['feature_units']
            if unit.alliance == _PLAYER_SELF
        ])
        return initial_obs, {}

    def step(self, action):
        sc2_action = self._transform_action(action, self.current_timestep)
        timesteps = self.env.step([sc2_action])
        self.current_timestep = timesteps[0]
        obs = self._extract_observation(timesteps[0])

        reward = self._calculate_reward(timesteps[0])
        done = self.current_timestep.last()
        truncated = False
        return obs, reward, done, truncated, {}

    def _calculate_reward(self, timestep):
        """Calcula a recompensa com base nos Zerglings derrotados e penaliza unidades aliadas destruídas."""
        player_score = timestep.observation['score_cumulative']
        reward = 0

        # Calcula a recompensa baseada em inimigos derrotados
        current_killed_units = player_score.score
        killed_zerglings = (current_killed_units - self.cumulative_kills)
        if killed_zerglings > 0:
            reward += killed_zerglings * 10
        self.cumulative_kills = current_killed_units

        # Penaliza unidades aliadas destruídas
        game_units = timestep.observation['feature_units']
        current_allied_units = [unit for unit in game_units if unit.alliance == _PLAYER_SELF]
        lost_units = self.total_allied_units - len(current_allied_units)

        if lost_units > 0:
            reward -= (lost_units * 50)
            self.total_allied_units = len(current_allied_units)

        return reward
    def _extract_observation(self, timestep):
        screen = np.array(timestep.observation['feature_screen'][self.features], dtype=np.uint8)
        return screen

    def _select_units(self, timestep, unit_mask):
        """Seleciona unidades com base em uma máscara binária."""
        game_units = timestep.observation['feature_units']
        units_self = [unit for unit in game_units if unit.alliance == _PLAYER_SELF]

        selected_units = []
        for i, select in enumerate(unit_mask):
            if select and len(units_self) > i:
                selected_units.append(units_self[i])

        if selected_units:
            for unit in selected_units:
                x = max(0, min(unit.x, 47))
                y = max(0, min(unit.y, 47))
                self.env.step([actions.FUNCTIONS.select_point("select", (x, y))])

        return actions.FUNCTIONS.no_op()

    def _transform_action(self, action, timestep):
        if timestep is None:
            raise ValueError("Timestep is not initialized.")

        unit_mask = action[:3]
        action_type, target_x, target_y = action[3:]
        available_actions = timestep.observation.available_actions

        target_x = max(0, min(target_x, 47))
        target_y = max(0, min(target_y, 47))

        if action_type == 0:  # Selecionar unidades
            return self._select_units(timestep, unit_mask)

        elif action_type == 1:  # Mover unidades para uma localização
            if actions.FUNCTIONS.Move_screen.id in available_actions:
                return actions.FUNCTIONS.Move_screen("now", (target_x, target_y))

        elif action_type == 2:  # Atacar em uma localização
            if actions.FUNCTIONS.Attack_screen.id in available_actions:
                return actions.FUNCTIONS.Attack_screen("now", (target_x, target_y))

        elif action_type == 3:  # Mover a câmera para uma localização
            if actions.FUNCTIONS.move_camera.id in available_actions:
                return actions.FUNCTIONS.move_camera((target_x, target_y))

        return actions.FUNCTIONS.no_op()

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()
