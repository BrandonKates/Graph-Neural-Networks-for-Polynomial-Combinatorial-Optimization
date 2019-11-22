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


def params(num_nodes):
  distances = np.random.random( (num_nodes, 2) )
  dist_mat = np.round( distance_matrix(distances, distances), 2 ).flatten()
  generated_weights = str(dist_mat[0])
  for i in range(1,dist_mat.size):
    generated_weights+="," + str(dist_mat[i])
    env_configs = {
       "num_nodes": num_nodes,
       "weights": generated_weights
       }
  return env_configs   

def spiel_params(num_nodes):
    args = params(num_nodes)
    return {k:pyspiel.GameParameter(v) for k,v in args.items()}
    
  
