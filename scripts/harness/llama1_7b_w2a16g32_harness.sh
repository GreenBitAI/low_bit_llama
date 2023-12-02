#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python llama_harness.py -s 7b -b 2 -g 32 -v 1
