#!/bin/bash

MDP="-"
ALGORITHM="-"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --mdp)
    MDP="$2"
    shift # past argument
    shift # past value
    ;;
    --algorithm)
    ALGORITHM="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}"

python3 planner.py --mdp "${MDP}" --algorithm "${ALGORITHM}"