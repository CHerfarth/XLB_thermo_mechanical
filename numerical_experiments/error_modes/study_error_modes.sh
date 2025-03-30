#!/bin/bash

epsilon=1.0
nodes_x=10
nodes_y=10
timesteps=1000
dt=0.0001
k=1
iterations=10

#for bookkeeping
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
#log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"
echo "All output logged in $log_file"
echo "Writing results to $results_file"

for ((i=0; i<iterations; i++))
do
    echo "--------------------"
    echo "Simulating with $nodes_x nodes and timestep of size $dt, # of timesteps: $timesteps     --->  epsilon = $epsilon"
    python3 error_mode_study.py $nodes_x $nodes_y $timesteps $dt $k $i"_"$results_file $i > tmp.txt

    #python3 plotter_single.py $results_file $dt $k".png"
    k=$(($k+1))
    echo "Iteration $i done"
done

final_file="final.csv"
cat "0_"$results_file > $final_file
for ((i=0; i<iterations; i++))
do
    csvjoin $final_file $i"_"$results_file > tmp.csv
    cat tmp.csv > $final_file
done
rm tmp.csv
rm *results*

python3 plotter.py $final_file $dt $iterations convergence_different_modes.png

#now do convergence study for one mode, different mesh sizes
k=2
nodes_x=4
nodes_y=4
iterations=5
for ((i=0; i<iterations; i++))
do
    echo "--------------------"
    echo "Simulating with $nodes_x nodes and timestep of size $dt, # of timesteps: $timesteps     --->  epsilon = $epsilon" 
    python3 error_mode_study.py $nodes_x $nodes_y $timesteps $dt $k $i"_"$results_file $i > tmp.txt
    nodes_x=$(($nodes_x*2))
    nodes_y=$(($nodes_y*2))
done

final_file="final.csv"
cat $"0_"$results_file > $final_file
for ((i=0; i<iterations; i++))
do
    csvjoin $final_file $i"_"$results_file > tmp.csv
    cat tmp.csv > $final_file
done
rm tmp.csv
rm *results*

python3 plotter.py $final_file $dt $iterations convergence_different_meshes.png





