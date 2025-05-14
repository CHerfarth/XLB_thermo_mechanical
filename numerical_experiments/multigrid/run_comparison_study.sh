#!/bin/bash

iterations=3
nodes_x=16
nodes_y=16
timesteps=6000
dt=0.01
for ((i=0; i<iterations; i++))
do
    python3 comparison.py $nodes_x $nodes_y $timesteps $dt #>> tmp.txt
    rm tmp.txt
    python3 plotter.py >> tmp.txt
    rm tmp.txt
    dir_name=plots_nodes_"$nodes_x"_dt_"$dt"
    mkdir $dir_name
    mv *png $dir_name


    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    dt=$(echo "$dt*0.25"|bc -l)
    timesteps=$((timesteps*4))


    echo "Iteration $i done"
done