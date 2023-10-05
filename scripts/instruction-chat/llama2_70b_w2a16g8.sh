#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python llama_2b_inference.py -s 70b-chat -v 2 -g 8

