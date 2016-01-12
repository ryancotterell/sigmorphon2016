#!/bin/bash

##############################################################
# SIGMORPHON shared task 2016                                #
# Script to run baseline on all tasks against training data  #
# and provide evaluation.                                    #
# -MH20151205                                                #
##############################################################

path="./../../data/"
mkdir -p $path"baseline"

for t in {1..3}
do
    for f in $path*-task1-train; do
	language=$(echo $f | sed s/-task1-train// | sed s/.*data.//)

	set -x
	./baseline.py --language=$language --task=$t --path=$path > $path"baseline/"$language"-task"$t"-out"
	./../evalm.py --golden $(echo $f | sed s/task1-train/task$t-dev/) --guesses $path"baseline/"$language"-task"$t"-out" > $path"baseline/"$language"-task"$t"-results"
	set +x
    done
done
