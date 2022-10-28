import random
import pyspiel
import numpy as np

# print(pyspiel.registered_names())
game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()

while not state.is_terminal():
  legal_actions = state.legal_actions()

  print(state.num_players())
  exit()

  if state.is_chance_node():
    # Sample a chance event outcome.
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(action)
  else:
    # The algorithm can pick an action based on an observation (fully observable
    # games) or an information state (information available for that player)
    # We arbitrarily select the first available action as an example.
    action = legal_actions[0]
    state.apply_action(action)
