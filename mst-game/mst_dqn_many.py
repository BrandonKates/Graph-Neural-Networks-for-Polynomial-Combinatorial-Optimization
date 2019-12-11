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

"""DQN agents trained on Minimum Spanning Tree by independent Q-learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
import mst_setup as mst

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent.")
#flags.DEFINE_integer("load_from_checkpoint", int(1), "If 1 then load from checkpoint")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

flags.DEFINE_integer(
    "test_every", 100000,
    "Episode frequency at which the DQN agents are fully tested on 10000 different graphs.")


# DQN model hyper-parameters
flags.DEFINE_string("game", "mst", "Name of the game")
flags.DEFINE_string("game_version", "easy", "Version of the Game")
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("num_nodes", 10, "Number of Nodes to Train On")

def save_rewards_as_csv(predicted_rewards, actual_rewards, epoch, num_nodes, game_version):
    df = pd.DataFrame([], columns=["Actual_Rewards", "Predicted_Rewards"])
    df.Actual_Rewards = actual_rewards
    df.Predicted_Rewards = predicted_rewards
    save_loc = "./dqn_results/"+str(game_version)+"/"+str(num_nodes)+"nodes/"+ "epoch" + str(epoch) +".csv"
    df.to_csv(save_loc, index=False)

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  cur_agents = trained_agents[0]
  for _ in range(num_episodes):
    time_step = env.reset()
    episode_rewards = 0
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = cur_agents.step(
         time_step, is_evaluation=True)
      action_list = [agent_output.action]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards[0]
        #print(time_step.rewards[player_pos])
    sum_episode_rewards += episode_rewards
  return sum_episode_rewards / num_episodes

def test_trained_bot(test_games, test_rewards, agent, epoch, num_nodes, game, game_version):
  test_games = [test_games[0]]
  test_rewards =[test_rewards[0]]

  sum_episode_rewards = 0
  cur_agents = agent
  num_episodes = len(test_games)
  all_episode_rewards = []
  for i in range(num_episodes):
    env = rl_environment.Environment(game, **test_games[i])
    time_step = env.reset() # (set new environment with random weights)
    actual_rewards  = test_rewards[i]
    episode_rewards = 0
    num_actions_before_cycle = 0
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = cur_agents.step(time_step, is_evaluation=True)
      action_list = [agent_output.action]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards[0]
      print("(Action, Reward): ", action_list[0], time_step.rewards[0])
      num_actions_before_cycle+=1
    print("Actual Rewards: ", actual_rewards)
    print("Episode Rewards: ", episode_rewards)
    print("Num Actions Before Cycle: ", num_actions_before_cycle)
    sum_episode_rewards += (actual_rewards - episode_rewards) # compute the distance away from expected MST
    all_episode_rewards.append(episode_rewards)
  #save_rewards_as_csv(all_episode_rewards, test_rewards, epoch, num_nodes, game_version)
  return sum_episode_rewards / num_episodes

def main(_):
  game = FLAGS.game # Set the game
  num_players = 1
  train_games, train_rewards, test_games, test_rewards = mst.game_params(FLAGS.num_nodes) # Load from files
  env_configs = train_games[0]
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = FLAGS.num_nodes * FLAGS.num_nodes * 3 #env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"] # number of possible actions

  print("Info State Size: ", info_state_size)
  print("Num Actions: ", num_actions)  
    
  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.125)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    agents = [
        dqn.DQN(
            session=sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    #saver = tf.train.import_meta_graph('/home/jupyter/ORIE-GNN-bjk224/mst-game/dqn_checkpoints/dqn_20epochs_mst_medium/dqn_test-399999.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('/home/jupyter/ORIE-GNN-bjk224/mst-game/dqn_checkpoints/dqn_20epochs_mst_medium/'))
    
    for ep in range(FLAGS.num_train_episodes):
      print(env_configs)
      #env_configs = train_games[ep % len(train_games)]
      #env = rl_environment.Environment(game, **env_configs)
      episode_reward = train_rewards[ep % len(train_games)]
      if (ep + 1) % FLAGS.eval_every == 0:
        r_mean = eval_against_random_bots(env, agents, random_agents, 0)
        logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
        #saver.save(sess, FLAGS.checkpoint_dir, ep)
        print("Actual MST Value: ", episode_reward)
      if (ep + 1) % FLAGS.test_every == 0:
        test_accuracy = test_trained_bot(test_games, test_rewards, agents[0], ep, FLAGS.num_nodes, game, FLAGS.game_version)
        logging.info("[%s] Test Accuracy: %s", ep + 1, test_accuracy)

      #env = rl_environment.Environment(game, **games[ep])
      time_step = env.reset()
      # print("TRAIN"+"*"*80)
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        #print("(Action, Reward): ", action_list[0], time_step.rewards[0])

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

    #print("Actual MST: ", train_rewards)
if __name__ == "__main__":
  app.run(main)
