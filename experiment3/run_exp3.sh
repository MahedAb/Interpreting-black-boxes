#!/bin/bash

cd experiment3

echo --------- script starts ---------------
echo

for bbox in MLP SVM
	do
		python3 src/wrapper_exp3.py --M 20 --s 4 --max_iter 20 --k 20 --lr 0.05 --bb $bbox
	done

echo ----------script ends -------------------