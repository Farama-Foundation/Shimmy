"""Environment configures."""

BSUITE_ENVS = (
    "bandit",
    "bandit_noise",
    "bandit_scale",
    "cartpole",
    "cartpole_noise",
    "cartpole_scale",
    "cartpole_swingup",
    "catch",
    "catch_noise",
    "catch_scale",
    "deep_sea",
    "deep_sea_stochastic",
    "discounting_chain",
    "memory_len",
    "memory_size",
    "mnist",
    "mnist_noise",
    "mnist_scale",
    "mountain_car",
    "mountain_car_noise",
    "mountain_car_scale",
    "umbrella_distract",
    "umbrella_length",
)

DM_CONTROL_SUITE_ENVS = (
    ("acrobot", "swingup"),
    ("acrobot", "swingup_sparse"),
    ("ball_in_cup", "catch"),
    ("cartpole", "balance"),
    ("cartpole", "balance_sparse"),
    ("cartpole", "swingup"),
    ("cartpole", "swingup_sparse"),
    ("cartpole", "two_poles"),
    ("cartpole", "three_poles"),
    ("cheetah", "run"),
    ("dog", "stand"),
    ("dog", "walk"),
    ("dog", "trot"),
    ("dog", "run"),
    ("dog", "fetch"),
    ("finger", "spin"),
    ("finger", "turn_easy"),
    ("finger", "turn_hard"),
    ("fish", "upright"),
    ("fish", "swim"),
    ("hopper", "stand"),
    ("hopper", "hop"),
    ("humanoid", "stand"),
    ("humanoid", "walk"),
    ("humanoid", "run"),
    ("humanoid", "run_pure_state"),
    ("humanoid_CMU", "stand"),
    ("humanoid_CMU", "walk"),
    ("humanoid_CMU", "run"),
    ("lqr", "lqr_2_1"),
    ("lqr", "lqr_6_2"),
    ("manipulator", "bring_ball"),
    ("manipulator", "bring_peg"),
    ("manipulator", "insert_ball"),
    ("manipulator", "insert_peg"),
    ("pendulum", "swingup"),
    ("point_mass", "easy"),
    ("point_mass", "hard"),
    ("quadruped", "walk"),
    ("quadruped", "run"),
    ("quadruped", "escape"),
    ("quadruped", "fetch"),
    ("reacher", "easy"),
    ("reacher", "hard"),
    ("stacker", "stack_2"),
    ("stacker", "stack_4"),
    ("swimmer", "swimmer6"),
    ("swimmer", "swimmer15"),
    ("walker", "stand"),
    ("walker", "walk"),
    ("walker", "run"),
)


DM_CONTROL_MANIPULATION_ENVS = (
    "stack_2_bricks_features",
    "stack_2_bricks_vision",
    "stack_2_bricks_moveable_base_features",
    "stack_2_bricks_moveable_base_vision",
    "stack_3_bricks_features",
    "stack_3_bricks_vision",
    "stack_3_bricks_random_order_features",
    "stack_2_of_3_bricks_random_order_features",
    "stack_2_of_3_bricks_random_order_vision",
    "reassemble_3_bricks_fixed_order_features",
    "reassemble_3_bricks_fixed_order_vision",
    "reassemble_5_bricks_random_order_features",
    "reassemble_5_bricks_random_order_vision",
    "lift_brick_features",
    "lift_brick_vision",
    "lift_large_box_features",
    "lift_large_box_vision",
    "place_brick_features",
    "place_brick_vision",
    "place_cradle_features",
    "place_cradle_vision",
    "reach_duplo_features",
    "reach_duplo_vision",
    "reach_site_features",
    "reach_site_vision",
)
