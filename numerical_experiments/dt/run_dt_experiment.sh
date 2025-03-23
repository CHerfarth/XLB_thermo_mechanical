#!/bin/bash

epsilon=1.0
nodes_x=50
nodes_y=50
timesteps=10000
post_process_interval=10
dt=0.1
iterations=8
applying_bc=0

#for bookkeeping
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"
echo "All output logged in $log_file"
echo "Applying BC: $applying_bc"
echo "dt,iteration,l2_disp,linf_disp,l2_stress,linf_stress" > $results_file

for ((i=0; i<iterations; i++))
do
    echo "--------------------"
    echo "Simulating with $nodes_x nodes and timestep of size $dt, # of timesteps: $timesteps     --->  epsilon = $epsilon"

    python3 dt_experiment.py $nodes_x $nodes_y $timesteps $dt $applying_bc $post_process_interval tmp_results.csv >  tmp_1.txt 
    cat tmp_1.txt >> $log_file #write to log

    #plot convergence
    python3 plotter_single.py tmp_results.csv $dt dt_${dt}_nodesx_${nodes_x}.png

    #get L2 disp error
    cat tmp_1.txt | grep "L2_disp" > tmp_2.txt
    error_L2_disp=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    #get Linf disp error
    cat tmp_1.txt | grep "Linf_disp" > tmp_2.txt
    error_Linf_disp=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    #get L2 stress
    cat tmp_1.txt | grep "L2_stress" > tmp_2.txt
    error_L2_stress=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
    
    #get Linf stress
    cat tmp_1.txt | grep "Linf_stress" > tmp_2.txt
    error_Linf_stress=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

    #get timesteps
    cat tmp_1.txt | grep "iteration" > tmp_2.txt
    iteration=$(cat tmp_2.txt | grep -oE '[0-9]+')


    echo "Error: $error_L2_disp, $error_Linf_disp, $error_L2_stress, $error_Linf_stress"
    echo "$dt,$iteration,$error_L2_disp, $error_Linf_disp, $error_L2_stress, $error_Linf_stress" >> $results_file
    rm tmp*

    #decrease expsilon
    dt=$(echo "$dt*0.25"|bc -l)
    #timesteps=$((timesteps*2))
    #post_process_interval=$((post_process_interval*2))

    echo "Iteration $i done"
done

python3 plotter_double.py $results_file total_results.png
