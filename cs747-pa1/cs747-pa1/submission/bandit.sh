#!/bin/bash

INSTANCE="-"
ALGORITHM="-"
RANDOMSEED="-"
EPSILON="-"
HORIZON="-"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --instance)
    INSTANCE="$2"
    shift # past argument
    shift # past value
    ;;
    --algorithm)
    ALGORITHM="$2"
    shift # past argument
    shift # past value
    ;;
    --randomSeed)
    RANDOMSEED="$2"
    shift # past argument
    shift # past value
    ;;
    --epsilon)
    EPSILON="$2"
    shift # past argument
    shift # past value
    ;;
    --horizon)
    HORIZON="$2"
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

python3 bandit.py --instance "${INSTANCE}" --algorithm "${ALGORITHM}" --randomSeed "${RANDOMSEED}" --epsilon "${EPSILON}" --horizon "${HORIZON}"