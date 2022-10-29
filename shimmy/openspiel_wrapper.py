"""Wrapper to convert a openspiel environment into a pettingzoo compatible environment."""

import functools
from typing import Dict, Optional

import numpy as np
import pettingzoo as pz
import pyspiel
from gymnasium import spaces
from gymnasium.spaces import space
from gymnasium.utils import seeding
from pettingzoo.utils.env import AgentID


class OpenspielWrapper(pz.AECEnv):
    """Wrapper that converts a openspiel environment into a pettingzoo environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        game: pyspiel.Game,
        render_mode: Optional[str],
    ):
        """Wrapper that converts a openspiel environment into a pettingzoo environment.

        Args:
            game (pyspiel.Game): game
            render_mode (Optional[str]): render_mode
        """
        self.game = game
        self.possible_agents = [
            "player_" + str(r) for r in range(self.game.num_players())
        ]
        self.agent_id_name_mapping = dict(
            zip(range(self.game.num_players()), self.possible_agents)
        )
        self.agent_name_id_mapping = dict(
            zip(self.possible_agents, range(self.game.num_players()))
        )

        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        """observation_space.

        Args:
            agent (AgentID): agent
        """
        try:
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.game.observation_tensor_shape(),
                dtype=np.float64,
            )
        except pyspiel.SpielError:
            return spaces.Text(max_length=2**16)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        """action_space.

        Args:
            agent (AgentID): agent
        """
        try:
            return spaces.Discrete(self.game.num_distinct_actions())
        except pyspiel.SpielError as e:
            raise NotImplementedError(f"{str(e)[:-1]} for {self.game}.")

    def render(self):
        """render."""
        raise NotImplementedError("No render available for openspiel.")

    def observe(self, agent: AgentID):
        """observe.

        Args:
            agent (AgentID): agent
        """
        return self.observations[agent]

    def close(self):
        """close."""
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: Optional[bool] = False,
        options: Optional[Dict] = None,
    ):
        """reset.

        Args:
            seed (Optional[int]): seed
            return_info (Optional[bool]): return_info
            options (Optional[Dict]): options
        """
        # initialize the seed
        self.np_random, seed = seeding.np_random(seed)

        # all agents
        self.agents = self.possible_agents[:]

        # boilerplate stuff
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # get a new game state, game_length = number of game nodes
        self.game_length = 1
        self.game_state = self.game.new_initial_state()

        # holders in case of simultaneous actions
        self.simultaneous_actions = dict()

        # step through chance nodes
        # then update obs and act masks
        # then choose next agent
        self._execute_chance_node()
        self._update_observations()
        self._update_action_masks()
        self._choose_next_agent()

    def _execute_chance_node(self):
        """_execute_chance_node.

        Some game states in the environment are out of the control of the agent.
        In these states, we need to sample the next state.
        There is also the possibility of multiple consecutive chance states, hence the `while`.
        """
        # if the game state is a chance node, choose a random outcome
        while self.game_state.is_chance_node():
            self.game_length += 1
            outcomes_with_probs = self.game_state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = self.np_random.choice(action_list, p=prob_list)
            self.game_state.apply_action(action)

    def _execute_action_node(self, action: int):
        """_execute_action_node.

        Advances the game state.
        We need to deal with 2 possible cases:
            - simultaneous game state where all the agents must step together
            - non-simultaneous game state where only one agent steps at a time

        To handle the simultaneous game state, we must step the environment a sufficient number of times,
        such that all actions for all agents have been collected before we can step the environment.

        To handle the non-simultaneous game state, we can just step the environment one agent at a time.

        Args:
            action (int): action
        """
        # if the game state is a simultaneous node, we need to collect all actions first
        if self.game_state.is_simultaneous_node():
            # store the agent's action
            self.simultaneous_actions[self.agent_selection] = action

            # set the agents reward to 0 since it's seen it
            self._cumulative_rewards[self.agent_selection] = 0

            # find agents for whom we don't have actions yet and get its action
            for agent in self.agents:
                if agent not in self.simultaneous_actions:
                    if np.sum(self.infos[agent]["action_mask"]) != 0:
                        self.agent_selection = agent
                        return
                    else:
                        # this will raise assertations with PZ api
                        self.simultaneous_actions[agent] = None

            # if we already have all the actions, just step regularly
            self.game_state.apply_actions(list(self.simultaneous_actions.values()))
            self.game_length += 1

            # clear the simultaneous actions holder
            self.simultaneous_actions = dict()
        else:
            # if not simultaneous, step the state generically
            self.game_state.apply_action(action)
            self.game_length += 1

            self._choose_next_agent()

    def _choose_next_agent(self):
        current_player = self.game_state.current_player()
        if current_player >= 0:
            # if it's a normal node, just select the next agent according to the node
            self.agent_selection = self.agent_id_name_mapping[current_player]
        else:
            # if not a normal node, we need to be careful to choose only valid agents
            for agent in self.agents:
                if np.sum(self.infos[agent]["action_mask"]) != 0:
                    self.agent_selection = agent

    def _update_observations(self):
        """Updates all the observations inside the observations dictionary."""
        if isinstance(self.observation_space(self.agents[0]), spaces.Box):
            self.observations = {
                a: np.array(
                    self.game_state.observation_tensor(self.agent_name_id_mapping[a])
                ).reshape(self.game.observation_tensor_shape())
                for a in self.agents
            }
        elif isinstance(self.observation_space(self.agents[0]), spaces.Text):
            self.observations = {
                a: self.game_state.observation_string(self.agent_name_id_mapping[a])
                for a in self.agents
            }

    def _update_action_masks(self):
        """Updates all the action masks inside the infos dictionary."""
        for agent_id in range(self.game.num_players()):
            agent_name = self.agent_id_name_mapping[agent_id]
            action_mask = np.zeros(self.action_space(agent_name).n, dtype=np.int8)

            try:
                action_mask[self.game_state.legal_actions(agent_id)] = 1
            except pyspiel.SpielError:
                pass

            self.infos[agent_name] = {"action_mask": action_mask}

    def _update_rewards(self):
        """Updates all the _cumulative_rewards of the environment."""
        # update cumulative rewards
        rewards = self.game_state.rewards()
        self._cumulative_rewards = {
            self.agent_id_name_mapping[id]: rewards[id]
            for id in range(self.game.num_players())
        }

    def _end_routine(self):
        """Method that handles the routines that happen at environment termination.

        Since all agents end together we can hack our way around it.
        """
        # check for terminal
        self.terminations = {a: False for a in self.agents}
        if self.game_state.is_terminal():
            self.terminations = {a: True for a in self.agents}

        # check for action masks because openspiel doesn't do it themselves
        action_mask_sum = 0
        for agent in self.agents:
            action_mask_sum += np.sum(self.infos[agent]["action_mask"])

        # if all actions are illegal for all agents, declare terminal
        if action_mask_sum == 0:
            self.terminations = {a: True for a in self.agents}

        # check for truncation
        self.truncations = {a: False for a in self.agents}
        if self.game_length > self.game.max_game_length():
            self.truncations = {a: True for a in self.agents}

        # if terminal, start deleting agents
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self.agents.remove(self.agent_selection)
            self._cumulative_rewards.pop(self.agent_selection)
            self.rewards.pop(self.agent_selection)
            self.terminations.pop(self.agent_selection)
            self.truncations.pop(self.agent_selection)
            self.infos.pop(self.agent_selection)

            if self.agents:
                self.agent_selection = self.agents[0]

            return True

        return False

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action
        """
        # handle the possibility of an end step
        if self._end_routine():
            return
        else:
            # step the environment
            self._execute_action_node(action)
            self._execute_chance_node()
            self._update_observations()
            self._update_action_masks()
            self._update_rewards()
