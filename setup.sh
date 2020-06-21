#!/bin/bash

# Nebula main directory
nebuladir=$PWD
# Nebula datasets directory
datasetdir=$nebuladir/datasets

# Create dataset lists.
for d in $datasetdir/*; do
	dataset=$(basename $d)
	echo -e "# Creating $dataset dataset lists ..."
	if [[ $dataset = 'mnist' ]]; then
		cd $d; eval find $d/test -name  \*.jpg > test.lst
	elif [[ $dataset = 'imagenet' ]]; then	
		cd $d; eval find $d/test -name  \*.JPEG > test.lst
	fi
done
