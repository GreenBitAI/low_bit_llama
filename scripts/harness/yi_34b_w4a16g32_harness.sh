#!/in/bash
CUDA_VISIBLE_DEVICES=0 python yi_harness.py -s 34b -b 4 -g 32 --batch_size 12 --tasks truthfulqa_mc #anli_r1,anli_r2 #,anli_r3,wic,rte,record
