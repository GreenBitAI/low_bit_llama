#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python llama_2b_evaluate.py -s 1.1b -v 2 -g 32
