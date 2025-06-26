gamma=1

E=0.2
nu=0.3
python3 smoothing_factor_single.py $E $nu $gamma

E=0.5
nu=0.5
python3 smoothing_factor_single.py $E $nu $gamma

E=0.8
nu=0.7
python3 smoothing_factor_single.py $E $nu $gamma

mkdir contours
mv *png contours
mv *eps contours
