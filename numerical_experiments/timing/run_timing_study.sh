#!/bin/bash

base_nodes_x=16
base_nodes_y=16
max_multi=5000
max_standard=500000

base_E=0.2
base_nu=0.3

d_E=0.2
num_E=3
d_nu=0.1
num_nu=4

iterations=9
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

repeat_iterations=5

timing_cutoff=40


nu=$base_nu
for ((l=0; l<num_nu; l++))
do
    E=$base_E
    for ((k=0; k<num_E; k++))
    do
        echo "dim,multigrid_converged,multigrid_time,multigrid_iterations,standard_converged,standard_time,standard_iterations,relaxed_converged,relaxed_time,relaxed_iterations" > $results_file
        for ((j=0; j<repeat_iterations; j++))
        do
            nodes_x=$base_nodes_x
            nodes_y=$base_nodes_y
            test_multigrid=1
            test_standard=0
            test_relaxed=0
            for ((i=0; i<iterations; i++))
            do
                python3 timing.py $nodes_x $nodes_y $max_multi $start_multi $interval_multi $max_standard $start_standard $intervals_standard $E $nu $test_multigrid $test_standard $test_relaxed >  tmp_1.txt 
                cat tmp_1.txt >> $log_file #write to log

                #get convergence of multigrid 
                cat tmp_1.txt | grep "Multigrid_Converged" > tmp_2.txt
                multigrid_converged=$(cat tmp_2.txt | grep -oE '[0-9]')
                #get runtime of multigrid 
                cat tmp_1.txt | grep "Multigrid_Time" > tmp_2.txt
                multigrid_time=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                #get iterations of multigrid 
                cat tmp_1.txt | grep "Multigrid_Iterations" > tmp_2.txt
                multigrid_iterations=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #dont test multigrid if running too long
                if (( $(echo "$multigrid_time > $timing_cutoff" | bc -l) )); then
                    test_multigrid=0
                fi

                #get convergence of standard 
                cat tmp_1.txt | grep "Standard_Converged" > tmp_2.txt
                standard_converged=$(cat tmp_2.txt | grep -oE '[0-9]')
                #get runtime of standard 
                cat tmp_1.txt | grep "Standard_Time" > tmp_2.txt
                standard_time=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                #get iterations of standard 
                cat tmp_1.txt | grep "Standard_Iterations" > tmp_2.txt
                standard_iterations=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #dont test standard if running too long
                if (( $(echo "$standard_time > $timing_cutoff" | bc -l) )); then
                    test_standard=0
                fi
                
                #get convergence of relaxed 
                cat tmp_1.txt | grep "Relaxed_Converged" > tmp_2.txt
                relaxed_converged=$(cat tmp_2.txt | grep -oE '[0-9]')
                #get runtime of relaxed 
                cat tmp_1.txt | grep "Relaxed_Time" > tmp_2.txt
                relaxed_time=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                #get iterations of relaxed 
                cat tmp_1.txt | grep "Relaxed_Iterations" > tmp_2.txt
                relaxed_iterations=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #dont test relaxed if running too long
                if (( $(echo "$relaxed_time > $timing_cutoff" | bc -l) )); then
                    test_relaxed=0
                fi

                echo "$nodes_x,$multigrid_converged,$multigrid_time,$multigrid_iterations,$standard_converged,$standard_time,$standard_iterations,$relaxed_converged,$relaxed_time,$relaxed_iterations" >> $results_file


                nodes_x=$((nodes_x*2))
                nodes_y=$((nodes_y*2))

                rm tmp*
            done
            
        done
        python3 plotter.py $results_file $E $nu
        mkdir plots 
        mkdir data
        mv runtimes.png plots/runtimes_E_"$E"_nu_"$nu".png
        mv multigrid.png plots/only_multi_E_"$E"_nu_"$nu".png
        mv $results_file data/results_E_"$E"_nu_"$nu".csv
        echo "Simulated with E $E and nu $nu"

        E=$(echo "$E+$d_E" | bc -l)
    done
    nu=$(echo "$nu+$d_nu" | bc -l)
done