#!/bin/bash

epsilon=1.0
nodes_x=20
nodes_y=20
timesteps=1000
dt=0.1
iterations=6

#for bookkeeping
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"
echo "All output logged in $log_file"
echo "Writing results to $results_file"
echo "Applying BC: $3"
echo "epsilon,error_L2_disp,error_Linf_disp,error_L2_stress,error_Linf_stress" > $results_file

for ((i=0; i<iterations; i++))
do
    echo "--------------------"
    echo "Simulating with $nodes_x nodes and timestep of size $dt, # of timesteps: $timesteps     --->  epsilon = $epsilon"

    python3 $1 $nodes_x $nodes_y $timesteps $dt $3 >  tmp_1.txt 
    cat tmp_1.txt >> $log_file #write to log

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

    echo "Error: $error_L2_disp, $error_Linf_disp, $error_L2_stress, $error_Linf_stress"
    echo "$epsilon,$error_L2_disp, $error_Linf_disp, $error_L2_stress, $error_Linf_stress" >> $results_file
    rm tmp*

    #decrease expsilon
    epsilon=$(echo "$epsilon*0.5" |bc -l)
    nodes_x=$((nodes_x*2))
    nodes_y=$((nodes_y*2))
    dt=$(echo "$dt*0.25"|bc -l)
    timesteps=$((timesteps*4))

    echo "Iteration $i done"
done

python3 $2 $results_file
