#!/usr/bin/env bash

STATE_DIR=/content/drive/My_Drive/output/
STATE_LOG=/content/drive/My_Drive/output/
DATA=/content/drive/Food-11/

if [ ! -d "$STATE_DIR" ]; then
    mkdir ${STATE_DIR}
fi


if [ ! -d "$STATE_LOG" ]; then
    mkdir ${STATE_LOG}
fi

cmd="python run_se.py -i DATA -o ${STATE_DIR} --epoch 20 --opt-func adam --lr 0.0001 --model-name se_default_food --plog ${STATE_LOG} --batch-size 40"
eval ${cmd}


