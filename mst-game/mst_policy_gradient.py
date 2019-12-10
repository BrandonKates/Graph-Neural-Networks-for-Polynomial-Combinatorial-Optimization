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

"""Policy gradient agents trained and evaluated on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
import mst_setup as mst
import numpy as np
import pandas as pd
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e6), "Number of train episodes.")
flags.DEFINE_integer("eval_every", int(1e4), "Eval agents every x episodes.")
flags.DEFINE_enum("loss_str", "rpg", ["rpg", "qpg", "rm"], "PG loss to use.")
flags.DEFINE_integer("num_nodes", 10, "Number of Nodes to Train On")
flags.DEFINE_string("game_version", "easy", "Version of the Game")
flags.DEFINE_integer(
    "test_every", 100000,
    "Episode frequency at which the DQN agents are fully tested on 10000 different graphs.")

class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._obs = {"info_state": [None], "legal_actions": [None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict

def save_rewards_as_csv(predicted_rewards, actual_rewards, epoch, num_nodes, game_version):
    df = pd.DataFrame([], columns=["Actual_Rewards", "Predicted_Rewards"])
    df.Actual_Rewards = actual_rewards
    df.Predicted_Rewards = predicted_rewards
    save_loc = "./policy_gradient_results/"+str(game_version)+"/"+str(num_nodes)+"nodes/"+ "epoch" + str(epoch) +".csv"
    df.to_csv(save_loc, index=False)
    
def test_trained_bot(test_games, test_rewards, agent, epoch, num_nodes, game, game_version):
  test_games = test_games[0:100]
  test_rewards = test_rewards[0:100]
  sum_episode_rewards = 0
  cur_agents = agent
  num_episodes = len(test_games)
  all_episode_rewards = []
  for i in range(num_episodes):
    env = rl_environment.Environment(game, **test_games[i])
    time_step = env.reset() # (set new environment with random weights)
    actual_rewards  = test_rewards[i]
    episode_rewards = 0
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = cur_agents.step(time_step, is_evaluation=True)
      action_list = [agent_output.action]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards[0]
  
    sum_episode_rewards += (actual_rewards - episode_rewards) # compute the distance away from expected MST
    all_episode_rewards.append(episode_rewards)
  save_rewards_as_csv(all_episode_rewards, test_rewards, epoch, num_nodes, game_version)
  return sum_episode_rewards / num_episodes

def main(_):
    
  game = "mst"
  num_players = 1
  train_games, train_rewards, test_games, test_rewards = mst.game_params(FLAGS.num_nodes)

  env_configs = train_games[0]
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = FLAGS.num_nodes * FLAGS.num_nodes * 3
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        policy_gradient.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=FLAGS.loss_str,
            hidden_layers_sizes=(128,)) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)

    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_episodes):
      env_configs = train_games[ep % len(train_games)]
      env = rl_environment.Environment(game, **env_configs)
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        #expl = exploitability.exploitability(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += "{}: {}\n".format(ep + 1, losses)#expl, losses)
        logging.info("%s", msg)
      if (ep + 1) % FLAGS.test_every == 0:
        test_accuracy = test_trained_bot(test_games, test_rewards, agents[0], ep, FLAGS.num_nodes, game,    FLAGS.game_version)
        logging.info("[%s] Test Accuracy: %s", ep + 1, test_accuracy)
        
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
