#!/bin/bash

base_nodes_x=16
base_nodes_y=16
max_multi=5000
max_standard=500000

base_E=.2
base_nu=.3

d_E=0.3
num_E=2
d_nu=0.3
num_nu=2

multigrid_dim=8
standard_dim=15

current_date_time="`date "+%Y-%m-%d_%H-%M-%S"`"
log_file="log_"$current_date_time".txt"
results_file="results_"$current_date_time".csv"

repeat_iterations=1 #20

timing_cutoff=40

nu=$base_nu
for ((l=0; l<num_nu; l++))
do
    E=$base_E
    for ((k=0; k<num_E; k++))
    do
        echo "dim,vcycle_converged_with_allocation,vcycle_time_with_allocation,vcycle_iterations_with_allocation,wcycle_converged_with_allocation,wcycle_time_with_allocation,wcycle_iterations_with_allocation,standard_converged_with_allocation,standard_time_with_allocation,standard_iterations_with_allocation,vcycle_converged_no_allocation,vcycle_time_no_allocation,vcycle_iterations_no_allocation,wcycle_converged_no_allocation,wcycle_time_no_allocation,wcycle_iterations_no_allocation,standard_converged_no_allocation,standard_time_no_allocation,standard_iterations_no_allocation,vcycle_wu,wcycle_wu,standard_wu" > $results_file
        for ((j=0; j<repeat_iterations; j++))
        do

            #---------------time multigrid (only powers of 2)--------------------
            nodes_x=$base_nodes_x
            nodes_y=$base_nodes_y
            test_multigrid=1
            test_standard=0
            for ((i=0; i<multigrid_dim; i++))
            do
                python3 timing.py $nodes_x $nodes_y $max_multi $max_standard $E $nu $test_multigrid $test_standard >  tmp_1.txt 
                cat tmp_1.txt >> $log_file #write to log


                #get convergence of vcycle with allocation
                cat tmp_1.txt | grep "VCycle_Converged_With_Allocation" > tmp_2.txt
                vcycle_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "VCycle_Time_With_Allocation" > tmp_2.txt
                vcycle_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "VCycle_Iterations_With_Allocation" > tmp_2.txt
                vcycle_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of vcycle no allocation
                cat tmp_1.txt | grep "VCycle_Converged_No_Allocation" > tmp_2.txt
                vcycle_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "VCycle_Time_No_Allocation" > tmp_2.txt
                vcycle_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "VCycle_Iterations_No_Allocation" > tmp_2.txt
                vcycle_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                cat tmp_1.txt | grep "VCycle_WU" > tmp_2.txt
                vcycle_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                #get convergence of wcycle with allocation
                cat tmp_1.txt | grep "WCycle_Converged_With_Allocation" > tmp_2.txt
                wcycle_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "WCycle_Time_With_Allocation" > tmp_2.txt
                wcycle_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "WCycle_Iterations_With_Allocation" > tmp_2.txt
                wcycle_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of wcycle no allocation
                cat tmp_1.txt | grep "WCycle_Converged_No_Allocation" > tmp_2.txt
                wcycle_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "WCycle_Time_No_Allocation" > tmp_2.txt
                wcycle_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "WCycle_Iterations_No_Allocation" > tmp_2.txt
                wcycle_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                cat tmp_1.txt | grep "WCycle_WU" > tmp_2.txt
                wcycle_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

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
                cat tmp_1.txt | grep "Standard_WU" > tmp_2.txt
                standard_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                echo "$nodes_x,$vcycle_converged_with_allocation,$vcycle_time_with_allocation,$vcycle_iterations_with_allocation,$wcycle_converged_with_allocation,$wcycle_time_with_allocation,$wcycle_iterations_with_allocation,$standard_converged_with_allocation,$standard_time_with_allocation,$standard_iterations_with_allocation,$vcycle_converged_no_allocation,$vcycle_time_no_allocation,$vcycle_iterations_no_allocation,$wcycle_converged_no_allocation,$wcycle_time_no_allocation,$wcycle_iterations_no_allocation,$standard_converged_no_allocation,$standard_time_no_allocation,$standard_iterations_no_allocation,$vcycle_wu,$wcycle_wu,$standard_wu" >> $results_file


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
            
            #-------------------------------time standard----------------------------------
            nodes_x=$base_nodes_x
            nodes_y=$base_nodes_y
            test_multigrid=0
            test_standard=1
            for ((i=0; i<standard_dim; i++))
            do
                python3 timing.py $nodes_x $nodes_y $max_multi $max_standard $E $nu $test_multigrid $test_standard >  tmp_1.txt 
                cat tmp_1.txt >> $log_file #write to log

                #get convergence of vcycle with allocation
                cat tmp_1.txt | grep "VCycle_Converged_With_Allocation" > tmp_2.txt
                vcycle_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "VCycle_Time_With_Allocation" > tmp_2.txt
                vcycle_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "VCycle_Iterations_With_Allocation" > tmp_2.txt
                vcycle_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of vcycle no allocation
                cat tmp_1.txt | grep "VCycle_Converged_No_Allocation" > tmp_2.txt
                vcycle_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "VCycle_Time_No_Allocation" > tmp_2.txt
                vcycle_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "VCycle_Iterations_No_Allocation" > tmp_2.txt
                vcycle_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                cat tmp_1.txt | grep "VCycle_WU" > tmp_2.txt
                vcycle_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                #get convergence of wcycle with allocation
                cat tmp_1.txt | grep "WCycle_Converged_With_Allocation" > tmp_2.txt
                wcycle_converged_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "WCycle_Time_With_Allocation" > tmp_2.txt
                wcycle_time_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "WCycle_Iterations_With_Allocation" > tmp_2.txt
                wcycle_iterations_with_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                #get convergence of wcycle no allocation
                cat tmp_1.txt | grep "WCycle_Converged_No_Allocation" > tmp_2.txt
                wcycle_converged_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]')
                cat tmp_1.txt | grep "WCycle_Time_No_Allocation" > tmp_2.txt
                wcycle_time_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')
                cat tmp_1.txt | grep "WCycle_Iterations_No_Allocation" > tmp_2.txt
                wcycle_iterations_no_allocation=$(cat tmp_2.txt | grep -oE '[0-9]+')
                cat tmp_1.txt | grep "WCycle_WU" > tmp_2.txt
                wcycle_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

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
                cat tmp_1.txt | grep "Standard_WU" > tmp_2.txt
                standard_wu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

                echo "$nodes_x,$vcycle_converged_with_allocation,$vcycle_time_with_allocation,$vcycle_iterations_with_allocation,$wcycle_converged_with_allocation,$wcycle_time_with_allocation,$wcycle_iterations_with_allocation,$standard_converged_with_allocation,$standard_time_with_allocation,$standard_iterations_with_allocation,$vcycle_converged_no_allocation,$vcycle_time_no_allocation,$vcycle_iterations_no_allocation,$wcycle_converged_no_allocation,$wcycle_time_no_allocation,$wcycle_iterations_no_allocation,$standard_converged_no_allocation,$standard_time_no_allocation,$standard_iterations_no_allocation,$vcycle_wu,$wcycle_wu,$standard_wu" >> $results_file

                nodes_x=$(printf "%.0f" $(echo "$nodes_x*1.3" | bc -l))
                nodes_y=$(printf "%.0f" $(echo "$nodes_y*1.3" | bc -l))

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
        for pdffile in *.pdf; do
            [ -e "$pdffile" ] && mv "$pdffile" "plots/${pdffile%.pdf}_E_${E}_nu_${nu}.pdf"
        done
        mv $results_file data/results_E_"$E"_nu_"$nu".csv
        echo "Simulated with E $E and nu $nu"

        E=$(echo "$E+$d_E" | bc -l)
    done
    nu=$(echo "$nu+$d_nu" | bc -l)
done