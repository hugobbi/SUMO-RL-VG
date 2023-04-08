#!/bin/sh

echo "-- NO VG --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py
done

echo "-- VG --"
echo "-- SIZE 1 --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py --vg_file ../dados_sim/dict_diamondtls15k250ajunction_s1.pkl
done

echo "-- SIZE 2 --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py --vg_file ../dados_sim/dict_diamondtls15k250ajunction_s2.pkl
done

echo "-- SIZE 3 --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py --vg_file ../dados_sim/dict_diamondtls15k250ajunction_s3.pkl
done

echo "-- SIZE 4 --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py --vg_file ../dados_sim/dict_diamondtls15k250ajunction_s4.pkl
done

echo "-- SIZE 5 --"
for i in $(seq 1 10);
do
    echo Run: $i
    python3 experiments/diamondTLsVG.py --vg_file ../dados_sim/dict_diamondtls15k250ajunction_s5.pkl
done
