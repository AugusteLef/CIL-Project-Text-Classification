#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../Data/train_pos_full_basic.txt ../Data/train_neg_full_basic.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
