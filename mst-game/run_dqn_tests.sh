#!/bin/bash

#python -u mst_dqn_many.py --num_nodes 4 --game=mst --eval_every 10000 --test_every 100000 --game_version=easy > "easy4node.out" 2> "easy4node.err" &
#python -u mst_dqn_many.py --num_nodes 4 --game=mst_medium --eval_every 10000 --test_every 100000 --game_version=medium > "medium4node.out" 2> "medium4node.err" &

#python -u mst_dqn_many.py --num_nodes 5 --game=mst --eval_every 10000 --test_every 100000 --game_version=easy > "easy5node.out" 2> "easy5node.err" &
#python -u mst_dqn_many.py --num_nodes 5 --game=mst_medium --eval_every 10000 --test_every 100000 --game_version=medium > "medium5node.out" 2> "medium5node.err" &

#python -u mst_dqn_many.py --num_nodes 10 --game=mst --eval_every 10000 --test_every 100000 --game_version=easy > "easy10node.out" 2> "easy10node.err" &
#python -u mst_dqn_many.py --num_nodes 10 --game=mst_medium --eval_every 10000 --test_every 100000 --game_version=medium > "medium10node.out" 2> "medium10node.err" &

python -u mst_dqn_many.py --num_nodes 20 --game=mst --eval_every 10000 --test_every 100000 --game_version=easy > "easy20node.out" 2> "easy20node.err" &
python -u mst_dqn_many.py --num_nodes 20 --game=mst_medium --eval_every 10000 --test_every 100000 --game_version=medium > "medium20node.out" 2> "medium20node.err" &