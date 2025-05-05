#!/bin/bash

nodes_x=16
nodes_y=16
dt=0.1
max_multi=5000
max_standard=500000

iterations=8
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

echo "dim,multigrid_converged,multigrid_time,standard_converged,standard_time" > $results_file

for ((i=0; i<iterations; i++))
do
    python3 timing.py $nodes_x $nodes_y $max_multi $start_multi $interval_multi $max_standard $start_standard $intervals_standard $dt > tmp_1.txt 
    cat tmp_1.txt >> $log_file #write to log

    #get convergence of multigrid 
    cat tmp_1.txt | grep "Multigrid_Converged" > tmp_2.txt
    multigrid_converged=$(cat tmp_2.txt | grep -oE '[0-9]')
    #get runtime of multigrid 
    cat tmp_1.txt | grep "Multigrid_Time" > tmp_2.txt
    multigrid_time=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
    #get convergence of standard 
    cat tmp_1.txt | grep "Standard_Converged" > tmp_2.txt
    standard_converged=$(cat tmp_2.txt | grep -oE '[0-9]')
    #get runtime of standard 
    cat tmp_1.txt | grep "Standard_Time" > tmp_2.txt
    standard_time=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    echo "$nodes_x,$multigrid_converged,$multigrid_time,$standard_converged,$standard_time" >> $results_file


    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    dt=$(echo "$dt*0.25"|bc -l)
done

python3 plotter.py $results_file