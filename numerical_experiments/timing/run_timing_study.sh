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

iterations=8
current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

repeat_iterations=1

timing_cutoff=40

nu=$base_nu
for ((l=0; l<num_nu; l++))
do
    E=$base_E
    for ((k=0; k<num_E; k++))
    do
        echo "grid_points,multigrid_converged_with_allocation,multigrid_time_with_allocation,multigrid_iterations_with_allocation,standard_converged_with_allocation,standard_time_with_allocation,standard_iterations_with_allocation,multigrid_converged_no_allocation,multigrid_time_no_allocation,multigrid_iterations_no_allocation,standard_converged_no_allocation,standard_time_no_allocation,standard_iterations_no_allocation" > $results_file
        for ((j=0; j<repeat_iterations; j++))
        do
            nodes_x=$base_nodes_x
            nodes_y=$base_nodes_y
            test_multigrid=1
            test_standard=1
            for ((i=0; i<iterations; i++))
            do
                python3 timing.py $nodes_x $nodes_y $max_multi $max_standard $E $nu $test_multigrid $test_standard >  tmp_1.txt 
                cat tmp_1.txt >> $log_file #write to log


                #get convergence of multigrid with allocation
                cat tmp_1.txt | grep "Multigrid_Converged_With_Allocation" > tmp_2.txt
                multigrid_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "Multigrid_Time_With_Allocation" > tmp_2.txt
                multigrid_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "Multigrid_Iterations_With_Allocation" > tmp_2.txt
                multigrid_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of multigrid no allocation
                cat tmp_1.txt | grep "Multigrid_Converged_No_Allocation" > tmp_2.txt
                multigrid_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "Multigrid_Time_No_Allocation" > tmp_2.txt
                multigrid_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "Multigrid_Iterations_No_Allocation" > tmp_2.txt
                multigrid_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of standard with allocation
                cat tmp_1.txt | grep "Standard_Converged_With_Allocation" > tmp_2.txt
                standard_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "Standard_Time_With_Allocation" > tmp_2.txt
                standard_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "Standard_Iterations_With_Allocation" > tmp_2.txt
                standard_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of standard no allocation
                cat tmp_1.txt | grep "Standard_Converged_No_Allocation" > tmp_2.txt
                standard_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "Standard_Time_No_Allocation" > tmp_2.txt
                standard_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "Standard_Iterations_No_Allocation" > tmp_2.txt
                standard_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')

                echo "$((nodes_x*nodes_x)),$multigrid_converged_with_allocation,$multigrid_time_with_allocation,$multigrid_iterations_with_allocation,$standard_converged_with_allocation,$standard_time_with_allocation,$standard_iterations_with_allocation,$multigrid_converged_no_allocation,$multigrid_time_no_allocation,$multigrid_iterations_no_allocation,$standard_converged_no_allocation,$standard_time_no_allocation,$standard_iterations_no_allocation" >> $results_file


                nodes_x=$((nodes_x*2))
                nodes_y=$((nodes_y*2))

                #dont test multigrid if running too long (no allocation)
                if [[ -n "$multigrid_time_no_allocation" ]] && (( $(echo "$multigrid_time_no_allocation > $timing_cutoff" | bc -l) )); then
                    test_multigrid=0
                fi
                #dont test standard if running too long (no allocation)
                if [[ -n "$standard_time_no_allocation" ]] && (( $(echo "$standard_time_no_allocation > $timing_cutoff" | bc -l) )); then
                    test_standard=0
                fi

                rm tmp*
            done
            
        done
        python3 plotter.py $results_file $E $nu
        mkdir plots 
        mkdir data
        for epsfile in *.eps; do
            [ -e "$epsfile" ] && mv "$epsfile" "plots/${epsfile%.eps}_E_${E}_nu_${nu}.eps"
        done
        for pngfile in *.png; do
            [ -e "$pngfile" ] && mv "$pngfile" "plots/${pngfile%.eps}_E_${E}_nu_${nu}.png"
        done
        mv $results_file data/results_E_"$E"_nu_"$nu".csv
        echo "Simulated with E $E and nu $nu"

        E=$(echo "$E+$d_E" | bc -l)
    done
    nu=$(echo "$nu+$d_nu" | bc -l)
done