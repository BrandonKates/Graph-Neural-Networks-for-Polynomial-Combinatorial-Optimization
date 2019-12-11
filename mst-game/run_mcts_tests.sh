





python -u mcts.py --num_nodes 5 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium5_5sim_10roll_node.out" 2> "mcts_results/medium5_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 5 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium5_3sim_3roll_node.out" 2> "mcts_results/medium5_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 10 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium10_5sim_10roll_node.out" 2> "mcts_results/medium10_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 10 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium10_3sim_3roll_node.out" 2> "mcts_results/medium10_3sim_3roll_node.err" &



python -u mcts.py --num_nodes 20 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium20_5sim_10roll_node.out" 2> "mcts_results/medium20_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 20 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium20_3sim_3roll_node.out" 2> "mcts_results/medium20_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 30 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium30_5sim_10roll_node.out" 2> "mcts_results/medium30_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 30 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium30_3sim_3roll_node.out" 2> "mcts_results/medium30_3sim_3roll_node.err" &



python -u mcts.py --num_nodes 5 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy5_5sim_10roll_node.out" 2> "mcts_results/easy5_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 5 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy5_3sim_3roll_node.out" 2> "mcts_results/easy5_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 10 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst--quiet > "mcts_results/easy10_5sim_10roll_node.out" 2> "mcts_results/easy10_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 10 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy10_3sim_3roll_node.out" 2> "mcts_results/easy10_3sim_3roll_node.err" &



python -u mcts.py --num_nodes 20 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy20_5sim_10roll_node.out" 2> "mcts_results/easy20_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 20 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy20_3sim_3roll_node.out" 2> "mcts_results/easy20_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 30 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy30_5sim_10roll_node.out" 2> "mcts_results/easy30_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 30 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy30_3sim_3roll_node.out" 2> "mcts_results/easy30_3sim_3roll_node.err" &



python -u mcts.py --num_nodes 40 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy40_5sim_10roll_node.out" 2> "mcts_results/easy40_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 40 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy40_3sim_3roll_node.out" 2> "mcts_results/easy40_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 40 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium40_5sim_10roll_node.out" 2> "mcts_results/medium40_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 40 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium40_3sim_3roll_node.out" 2> "mcts_results/medium40_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 50 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy50_5sim_10roll_node.out" 2> "mcts_results/easy50_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 50 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy50_3sim_3roll_node.out" 2> "mcts_results/easy50_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 50 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium50_5sim_10roll_node.out" 2> "mcts_results/medium50_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 50 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium50_3sim_3roll_node.out" 2> "mcts_results/medium50_3sim_3roll_node.err" &


python -u mcts.py --num_nodes 100 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=easy --game=mst --quiet > "mcts_results/easy100_5sim_10roll_node.out" 2> "mcts_results/easy100_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 100 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=easy --game=mst --quiet > "mcts_results/easy100_3sim_3roll_node.out" 2> "mcts_results/easy100_3sim_3roll_node.err" &

python -u mcts.py --num_nodes 100 --num_games 1000 --max_simulations 5 --rollout_count 10 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium100_5sim_10roll_node.out" 2> "mcts_results/medium100_5sim_10roll_node.err" &
python -u mcts.py --num_nodes 100 --num_games 1000 --max_simulations 3 --rollout_count 3 --game_version=medium --game=mst_medium --quiet > "mcts_results/medium100_3sim_3roll_node.out" 2> "mcts_results/medium100_3sim_3roll_node.err" &
