#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python yi_evaluate.py -s 6b -b 4 -g 32 -c
