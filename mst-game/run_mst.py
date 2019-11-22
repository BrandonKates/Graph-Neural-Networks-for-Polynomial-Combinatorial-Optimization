# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
import numpy as np
from scipy.spatial import distance_matrix

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "mst", "Name of the game")
flags.DEFINE_integer("num_nodes", None, "Number of nodes")
flags.DEFINE_string("load_state", None,
                    "A file containing a string to load a specific state")


def main(_):
  action_string = None

  print("Creating game: " + FLAGS.game)
  if FLAGS.num_nodes is not None:

    distances = np.random.random((FLAGS.num_nodes,2))
    dist_mat = np.round(distance_matrix(distances, distances),2).flatten()
    generated_weights = str(dist_mat[0])
    for i in range(1,dist_mat.size):
      generated_weights+="," + str(dist_mat[i])

    game = pyspiel.load_game(FLAGS.game,
                             {"num_nodes": pyspiel.GameParameter(FLAGS.num_nodes),
                              "weights": pyspiel.GameParameter(generated_weights)})
  else:
    game = pyspiel.load_game(FLAGS.game, {"num_nodes": pyspiel.GameParameter(5),
                                          "weights": pyspiel.GameParameter("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")})

  # Get a new state
  if FLAGS.load_state is not None:
    # Load a specific state
    state_string = ""
    with open(FLAGS.load_state, encoding="utf-8") as input_file:
      for line in input_file:
        state_string += line
    state_string = state_string.rstrip()
    print("Loading state:")
    print(state_string)
    print("")
    state = game.deserialize_state(state_string)
  else:
    state = game.new_initial_state()

  # Print the initial state
  print(str(state))

  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node

    legal_actions = state.legal_actions(state.current_player())
    print("Legal Actions: ", [(i//FLAGS.num_nodes, i%FLAGS.num_nodes) for i in legal_actions])
    # Decision node: sample action for the single current player
    action = random.choice(legal_actions)
    action_string = state.action_to_string(state.current_player(), action)
    print("Player ", state.current_player(), ", randomly sampled action: ",
          action_string)
    state.apply_action(action)

    print(str(state))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))

if __name__ == "__main__":
  app.run(main)
