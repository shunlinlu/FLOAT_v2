import os
import sys

cmd1 = "python OAT_sparse.py --ds 'cifar10' --b 128 -e 200 --lr 0.1 --use2BN --density 1\
     --prune 'magnitude' --growth 'momentum' --gpu 1 --seed 100"
os.system(cmd1)
