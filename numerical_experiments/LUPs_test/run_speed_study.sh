#!/bin/bash

nodes_x_base=16
nodes_y_base=16
dt_base=0.01
timesteps=10000

iterations=20
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

repeated_measurements=1

echo "dim,single_precision_periodic,single_precision_dirichlet,double_precision_periodic,double_precision_dirichlet" > $results_file


for ((k=0; k<repeated_measurements; k++))
do
    nodes_x=$nodes_x_base
    nodes_y=$nodes_y_base
    dt=$dt_base
    for ((i=0; i<iterations; i++))
    do
        single_precision=0
        python3 speed_tester.py $nodes_x $nodes_y $timesteps $dt $single_precision> tmp_1.txt 
        cat tmp_1.txt >> $log_file #write to log
        cat tmp_1.txt | grep "Periodic" > tmp_2.txt
        double_precision_periodic=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
        cat tmp_1.txt | grep "Dirichlet" > tmp_2.txt
        double_precision_dirichlet=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

        single_precision=1
        python3 speed_tester.py $nodes_x $nodes_y $timesteps $dt $single_precision> tmp_1.txt 
        cat tmp_1.txt >> $log_file #write to log
        cat tmp_1.txt | grep "Periodic" > tmp_2.txt
        single_precision_periodic=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
        cat tmp_1.txt | grep "Dirichlet" > tmp_2.txt
        single_precision_dirichlet=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

        echo "$nodes_x,$single_precision_periodic,$single_precision_dirichlet,$double_precision_periodic,$double_precision_dirichlet" >> $results_file


        nodes_x=$(printf "%.0f" $(echo "$nodes_x*1.3" | bc -l))
        nodes_y=$(printf "%.0f" $(echo "$nodes_y*1.3" | bc -l))
        #nodes_x=$((nodes_x+100))
        #nodes_y=$((nodes_y+100))
        dt=$(echo "$dt*0.25"|bc -l)
    done
done

python3 plotter.py $results_file
rm tmp*
