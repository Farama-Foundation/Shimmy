"""Environment configures."""

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

ALL_ATARI_GAMES = (
    "adventure",
    "air_raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "atlantis2",
    "backgammon",
    "bank_heist",
    "basic_math",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "blackjack",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "casino",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "crossbow",
    "darkchambers",
    "defender",
    "demon_attack",
    "donkey_kong",
    "double_dunk",
    "earthworld",
    "elevator_action",
    "enduro",
    "entombed",
    "et",
    "fishing_derby",
    "flag_capture",
    "freeway",
    "frogger",
    "frostbite",
    "galaxian",
    "gopher",
    "gravitar",
    "hangman",
    "haunted_house",
    "hero",
    "human_cannonball",
    "ice_hockey",
    "jamesbond",
    "journey_escape",
    "kaboom",
    "kangaroo",
    "keystone_kapers",
    "king_kong",
    "klax",
    "koolaid",
    "krull",
    "kung_fu_master",
    "laser_gates",
    "lost_luggage",
    "mario_bros",
    "miniature_golf",
    "montezuma_revenge",
    "mr_do",
    "ms_pacman",
    "name_this_game",
    "othello",
    "pacman",
    "phoenix",
    "pitfall",
    "pitfall2",
    "pong",
    "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "sir_lancelot",
    "skiing",
    "solaris",
    "space_invaders",
    "space_war",
    "star_gunner",
    "superman",
    "surround",
    "tennis",
    "tetris",
    "tic_tac_toe3_d",
    "time_pilot",
    "trondead",
    "turmoil",
    "tutankham",
    "up_n_down",
    "venture",
    "video_checkers",
    "video_pinball",
    "videochess",
    "videocube",
    "wizard_of_wor",
    "word_zapper",
    "yars_revenge",
    "zaxxon",
)
LEGACY_ATARI_GAMES = (
    "adventure",
    "air_raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank_heist",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper_command",
    "crazy_climber",
    "defender",
    "demon_attack",
    "double_dunk",
    "elevator_action",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    "journey_escape",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "ms_pacman",
    "name_this_game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space_invaders",
    "star_gunner",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "venture",
    "video_pinball",
    "wizard_of_wor",
    "yars_revenge",
    "zaxxon",
)
