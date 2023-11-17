#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python codellama_2b_inference.py -s instruction-34B -v 2 -g 8
