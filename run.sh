#!/bin/bash

# PARAMETERS
ROOTPATH="."
HS="256"
ES="32"
GS="512 512 512 512"
FS="512 1024 29"
EP=30
BS=64
SEED=2111192
PATIANCE=8
DECAYRATE=0.9
LR=1e-4
GC=60.0
OPTIM="Adam"
PDROP="0.0 0.2 0.0"
VALIDATEPER=20000
EXPDATE=`date "+day_%m_%d_%y-time_%H_%M_%S"`
BESTMODEL=${ROOTPATH}/saved_models/bestmodel_${EXPDATE}.jld2
LOGFILE=${ROOTPATH}/saved_models/logfile_${EXPDATE}.out
DATAPATH="data/processed"
SCENEPATH="data/CLEVR_v1.0/scenes"

# To print the command
echo ${ROOTPATH}/src/relnet.jl --logfile ${LOGFILE}\
      --bestmodel ${BESTMODEL}\
      --datapath ${DATAPATH}\
      --scenepath ${SCENEPATH}\
      --epoch ${EP}\
      --hidden ${HS}\
      --embed ${ES}\
      --batchsize ${BS}\
      --seed ${SEED}\
      --pdrop ${PDROP}\
      --gclip ${GC}\
      --lr ${LR}\
      --optim ${OPTIM}\
      --patiance ${PATIANCE}\
      --decayrate ${DECAYRATE}\
      --gs ${GS}\
      --fs ${FS}\
      --validate_per ${VALIDATEPER}

# START EXPERIMENT
julia ${ROOTPATH}/src/relnet.jl --logfile ${LOGFILE}\
      --bestmodel ${BESTMODEL}\
      --datapath ${DATAPATH}\
      --scenepath ${SCENEPATH}\
      --epoch ${EP}\
      --hidden ${HS}\
      --embed ${ES}\
      --batchsize ${BS}\
      --seed ${SEED}\
      --pdrop ${PDROP}\
      --gclip ${GC}\
      --lr ${LR}\
      --optim ${OPTIM}\
      --patiance ${PATIANCE}\
      --decayrate ${DECAYRATE}\
      --gs ${GS}\
      --fs ${FS}\
      --validate_per ${VALIDATEPER}
