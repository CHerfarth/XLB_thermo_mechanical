#!/bin/bash

epsilon=1.0
nodes_x=5
nodes_y=5
timesteps=1000000
dt=0.1
iterations=8

#for bookkeeping
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"
echo "All output logged in $log_file"
echo "Writing results to $results_file"
echo "epsilon,error" > $results_file

for ((i=0; i<iterations; i++))
do
    echo "--------------------"
    echo "Simulating with $nodes_x nodes and timestep of size $dt     --->  epsilon = $epsilon"

    python3 convergence_study.py $nodes_x $nodes_y $timesteps $dt >  tmp_1.txt
    cat tmp_1.txt >> $log_file #write to log
    cat tmp_1.txt | grep "Final" > tmp_2.txt
    error=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    echo "Error: $error"
    echo "$epsilon,$error" >> $results_file
    rm tmp*

    #decrease expsilon
    epsilon=$(echo "$epsilon*0.5" |bc -l)
    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    #dt=$(echo "$dt*0.25"|bc -l)

    echo "Iteration $i done"
done

python3 plotter.py $results_file