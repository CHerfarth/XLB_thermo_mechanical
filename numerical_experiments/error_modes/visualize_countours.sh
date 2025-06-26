E_list=(0.2 0.5 0.8)
nu_list=(0.3 0.5 0.7)

gamma=1
for i in "${!E_list[@]}"; do
    E=${E_list[$i]}
    nu=${nu_list[$i]}
    python3 smoothing_factor_single.py $E $nu $gamma
done

mkdir -p contours_standard
mv *pdf contours_standard

gamma=0.8
for i in "${!E_list[@]}"; do
    E=${E_list[$i]}
    nu=${nu_list[$i]}
    python3 smoothing_factor_single.py $E $nu $gamma
done

mkdir -p contours_relaxed
mv *pdf contours_relaxed
