#!/in/bash
CUDA_VISIBLE_DEVICES=0 python yi_harness.py -s 34b -b 2 -g 8 --tasks openbookqa,arc_easy,winogrande #,hellaswag,arc_challenge,boolq,race,truthfulqa_mc,anli_r3,wic,record
