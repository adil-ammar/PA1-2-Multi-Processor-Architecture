#!/bin/bash

for ((i = 1; i <= 32; i++))
do
    #echo "Number of threads is $i"
    ./CSCOpenMP matrix3.txt 8 $i
done