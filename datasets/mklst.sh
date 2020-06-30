#!/bin/bash

# Nebula datasets directory
datasetdir=`dirname $0`
cd $datasetdir
datasetdir=$PWD

# Create dataset lists.
for d in $datasetdir/*; do
    if [ -d $d ]; then
        dataset=$(basename $d)
        echo -e "# Creating $dataset dataset lists ..."
        if [[ $dataset = 'mnist' ]]; then
            cd $d; eval find $d/test -name  \*.jpg > test.lst
        elif [[ $dataset = 'imagenet' ]]; then	
            cd $d; eval find $d/test -name  \*.JPEG > test.lst
        fi
    fi
done
