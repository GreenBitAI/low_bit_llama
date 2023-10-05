#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python llama_2b_evaluate.py -s 30b -v 1 -g 8

