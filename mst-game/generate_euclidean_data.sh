#!/bin/bash
mkdir ./data/

# Train
python generate_graphs.py --num_graphs 10000 --num_nodes 10 --filename ./data/MST10_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 20 --filename ./data/MST20_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 30 --filename ./data/MST30_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 40 --filename ./data/MST40_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 50 --filename ./data/MST50_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 60 --filename ./data/MST60_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 70 --filename ./data/MST70_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 80 --filename ./data/MST80_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 90 --filename ./data/MST90_Euclidean_train.txt
python generate_graphs.py --num_graphs 10000 --num_nodes 100 --filename ./data/MST100_Euclidean_train.txt


python generate_graphs.py --num_graphs 2000 --num_nodes 10 --filename ./data/MST10_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 20 --filename ./data/MST20_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 30 --filename ./data/MST30_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 40 --filename ./data/MST40_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 50 --filename ./data/MST50_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 60 --filename ./data/MST60_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 70 --filename ./data/MST70_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 80 --filename ./data/MST80_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 90 --filename ./data/MST90_Euclidean_val.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 100 --filename ./data/MST100_Euclidean_val.txt

python generate_graphs.py --num_graphs 2000 --num_nodes 10 --filename ./data/MST10_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 20 --filename ./data/MST20_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 30 --filename ./data/MST30_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 40 --filename ./data/MST40_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 50 --filename ./data/MST50_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 60 --filename ./data/MST60_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 70 --filename ./data/MST70_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 80 --filename ./data/MST80_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 90 --filename ./data/MST90_Euclidean_test.txt
python generate_graphs.py --num_graphs 2000 --num_nodes 100 --filename ./data/MST100_Euclidean_test.txt