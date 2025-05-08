#!/bin/bash

nodes_x=512
nodes_y=512
timesteps=100
dt=0.00001


python3 convergence_tester.py $nodes_x $nodes_y $timesteps $dt > tmp_1.txt
cat tmp_1.txt > log.txt


cat tmp_1.txt | grep "E_scaled" > tmp_2.txt
E_scaled=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

cat tmp_1.txt | grep "nu" > tmp_2.txt
nu=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

echo "Finished Lattice Boltzmann"
echo "Simulated with E $E_scaled and nu $nu"

#get amplification factor
python3 ../error_modes/smoothing_factor_single.py $E_scaled $nu > tmp_1.txt

cat tmp_1.txt | grep "Amplification" > tmp_2.txt
amplification=$(cat tmp_2.txt | grep -oE '[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?')

echo "Expected amplification factor $amplification"

python3 plotter.py $amplification 6

echo "Plots generated"

rm tmp*



