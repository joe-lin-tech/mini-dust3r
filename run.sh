for (( j=0; j<=1900; j+=100 ))
do
    python mini_dust3r/api/inference.py --root_path data/EMDB2/09_outdoor_walk --start_frame $j
done