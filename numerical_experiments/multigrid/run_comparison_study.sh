#!/bin/bash

iterations=6
nodes_x=16
nodes_y=16
timesteps=200
E=0.2
nu=0.5

for ((i=0; i<iterations; i++))
do
    python3 comparison.py $nodes_x $nodes_y $timesteps $E $nu #>> tmp.txt
    rm tmp.txt
    python3 plotter.py >> tmp.txt
    rm tmp.txt
    dir_name=plots_nodes_"$nodes_x"_dt_"$dt"
    mkdir $dir_name
    mv *png $dir_name


    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    timesteps=$((timesteps*4))


    echo "Iteration $i done"
done