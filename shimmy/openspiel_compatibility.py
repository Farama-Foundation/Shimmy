"""Wrapper to convert a openspiel environment into a pettingzoo compatible environment."""
from __future__ import annotations

import functools
from typing import Dict, Optional

import numpy as np
import pettingzoo as pz
import pyspiel
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import AgentID


class OpenspielCompatibilityV0(pz.AECEnv):
    """This compatibility wrapper that converts an openspiel environment into a pettingzoo environment.

    OpenSpiel is a collection of environments and algorithms for research in general reinforcement learning
    and search/planning in games. OpenSpiel supports n-player (single- and multi- agent) zero-sum,
    cooperative and general-sum, one-shot and sequential, strictly turn-taking and simultaneous-move,
    perfect and imperfect information games, as well as traditional multiagent environments such as
    (partially- and fully- observable) grid worlds and social dilemmas.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        game: pyspiel.Game,
        render_mode: str | None,
    ):
        """Wrapper that converts a openspiel environment into a pettingzoo environment.

        Args:
            game (pyspiel.Game): game
            render_mode (Optional[str]): render_mode
        """
        super().__init__()
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

        self.game_type = self.game.get_type()

        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        """observation_space.

        We get the observation space from the underlying game.
        OpenSpiel possibly provides information and observation in several forms.
        This wrapper chooses which one to use depending on the following precedence:
            1. Observation Tensor
            2. Information Tensor
            3. Observation String
            4. Information String

        Args:
            agent (AgentID): agent
        """
        if self.game_type.provides_observation_tensor:
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.game.observation_tensor_shape(),
                dtype=np.float64,
            )
        elif self.game_type.provides_information_state_tensor:
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.game.information_state_tensor_shape(),
                dtype=np.float64,
            )
        elif (
            self.game_type.provides_information_state_string
            or self.game_type.provides_observation_string
        ):
            return spaces.Text(max_length=2**16)
        else:
            raise NotImplementedError(
                f"No information/observation tensor/string implemented for {self.game}."
            )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        """action_space.

        Args:
            agent (AgentID): agent
        """
        try:
            return spaces.Discrete(self.game.num_distinct_actions())
        except pyspiel.SpielError as e:
            raise NotImplementedError(
                f"{str(e)[:-1]} for action space for {self.game}."
            )

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
        seed: int | None = None,
        return_info: bool | None = False,
        options: dict | None = None,
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
        self.agent_ids = [self.agent_name_id_mapping[a] for a in self.agents]

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
        self._update_action_masks()
        self._update_observations()
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

            if all(a in self.simultaneous_actions for a in self.agents):
                # if we already have all the actions, just step regularly
                self.game_state.apply_actions(list(self.simultaneous_actions.values()))
                self.game_length += 1

                # clear the simultaneous actions holder
                self.simultaneous_actions = dict()
        else:
            # if not simultaneous, step the state generically
            self.game_state.apply_action(action)
            self.game_length += 1

    def _choose_next_agent(self):
        # handle possibility that we don't have anymore agents
        if not self.agents:
            return

        # handle terminal state
        if any(self.terminations.values()) or any(self.truncations.values()):
            # if terminal, choose the next valid agent
            if self.agents:
                self.agent_selection = self.agents[0]
            return

        # handle possibility for chance node
        if self.game_state.is_chance_node():
            # do nothing if chance node, we should not have gotten here
            raise Exception(
                "We should never have reached a point where we need to pick an agent on a chance node."
            )

        # handle possibility of simultaneous node
        if self.game_state.is_simultaneous_node():
            # find agents for whom we don't have actions yet if simultaneous node
            for agent in self.agents:
                if agent not in self.simultaneous_actions:
                    if np.sum(self.infos[agent]["action_mask"]) != 0:
                        self.agent_selection = agent
                        return
                    else:
                        # ignore agents where there are no valid actions
                        # this will raise assertations with PZ api
                        self.simultaneous_actions[agent] = None
            return

        # if we reached here, this is a normal node
        self.agent_selection = self.agent_id_name_mapping[
            self.game_state.current_player()
        ]

    def _update_observations(self):
        """Updates all the observations inside the observations dictionary."""
        if self.game_state.is_terminal():
            return

        if self.game_type.provides_observation_tensor:
            self.observations = {
                self.agents[a]: np.array(self.game_state.observation_tensor(a)).reshape(
                    self.observation_space(self.agents[0]).shape
                )
                for a in self.agent_ids
            }
        elif self.game_type.provides_information_state_tensor:
            self.observations = {
                self.agents[a]: np.array(
                    self.game_state.information_state_tensor(a)
                ).reshape(self.observation_space(self.agents[0]).shape)
                for a in self.agent_ids
            }
        elif self.game_type.provides_observation_string:
            self.observations = {
                self.agents[a]: self.game_state.observation_string(a)
                for a in self.agent_ids
            }
        elif self.game_type.provides_information_state_string:
            self.observations = {
                self.agents[a]: self.game_state.information_state_string(a)
                for a in self.agent_ids
            }
        else:
            raise NotImplementedError(
                f"No information/observation tensor/string implemented for {self.game}."
            )

    def _update_action_masks(self):
        """Updates all the action masks inside the infos dictionary."""
        for agent_id, agent_name in zip(self.agent_ids, self.agents):
            action_mask = np.zeros(self.action_space(agent_name).n, dtype=np.int8)
            action_mask[self.game_state.legal_actions(agent_id)] = 1

            self.infos[agent_name] = {"action_mask": action_mask}

    def _update_rewards(self):
        """Updates all the _cumulative_rewards of the environment."""
        # update cumulative rewards
        rewards = self.game_state.rewards()
        self._cumulative_rewards = {
            self.agent_id_name_mapping[id]: rewards[id] for id in self.agent_ids
        }

    def _update_termination_truncation(self):
        """Updates all terminations and truncations of the environment."""
        # check for terminal
        self.terminations = {a: self.terminations[a] for a in self.agents}
        if self.game_state.current_player() <= -4:
            self.terminations = {a: True for a in self.agents}

        # check for action masks because openspiel doesn't do it themselves
        action_mask_sum = 0
        for agent in self.agents:
            action_mask_sum += np.sum(self.infos[agent]["action_mask"])

        # if all actions are illegal for all agents, declare terminal
        if action_mask_sum == 0:
            self.terminations = {a: True for a in self.agents}

        # check for truncation
        self.truncations = {a: self.truncations[a] for a in self.agents}
        if self.game_length > self.game.max_game_length():
            self.truncations = {a: True for a in self.agents}

    def _end_routine(self):
        """Method that handles the routines that happen at environment termination.

        Since all agents end together we can hack our way around it.
        """
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

            return True

        return False

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action
        """
        # handle the possibility of an end step
        if not self._end_routine():
            # step the environment
            self._execute_action_node(action)
            self._execute_chance_node()
            self._update_action_masks()
            self._update_observations()
            self._update_rewards()
            self._update_termination_truncation()

        self._choose_next_agent()
