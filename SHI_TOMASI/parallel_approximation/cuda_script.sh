#!/bin/bash 

image_dimensions=(1024 2048 4096 7680 10240) 
sigma_values=(1.1) 
block_size=(8 16 32)

echo "image_size,block_size,comp_time" >> GPU.csv
for i in ${!image_dimensions[@]}; do 
    for j in ${!sigma_values[@]}; do 
        for k in ${!block_size[@]}; do
            for ((m=0;m<30;m++)) do
                ./approximate_shi_tomasi ~/dev/lenna/Lenna_org_${image_dimensions[$i]}.pgm ${sigma_values[$j]} 4 ${block_size[$k]} >> GPU.csv
            done
        done
    done
done
