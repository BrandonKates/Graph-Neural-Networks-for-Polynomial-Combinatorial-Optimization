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

import logging
from absl import app
from absl import flags
import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
import pyspiel
import generate_graphs as graphgen


def game_params(num_games, num_nodes):
  #distances = np.random.random( (num_nodes, 2) )
  #dist_mat = np.round( distance_matrix(distances, distances), 2 ).flatten()
  #generated_weights = str(dist_mat[0])
  game_data = get_game_data(num_games, num_nodes)
  game_configs = []
  game_rewards = []
  for i in range(num_games):
    env_configs = {
        "num_nodes": game_data['num_nodes'],
        "weights": game_data['inputs'][i]
      }
    game_configs.append(env_configs)
    game_rewards.append(game_data['rewards'][i])
  return game_configs, game_rewards

def spiel_params(num_games, num_nodes):
    configs, rewards = game_params(num_games, num_nodes)
    env_configs = [{k:pyspiel.GameParameter(v) for k,v in args.items()} for args in configs]
    return env_configs, rewards
    

def get_game_data(num_graphs, num_nodes):
    #args_list = ['--num_graphs',str(num_graphs),'--num_nodes',str(num_nodes)]
    return graphgen.generate_game_data(num_graphs, num_nodes)
    

# Would like to load in `num_graphs` graphs of `num_nodes` nodes of type `graph_type` 
