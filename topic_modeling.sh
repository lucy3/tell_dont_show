#!/bin/bash 
set -e

PREFIX=describe_gpt-4o_0 # name of your experiment, like passage_type in preprocess_text_for_lda.py

/mnt/data0/lucy/Mallet-202108/bin/mallet import-file --input /mnt/data0/lucy/literary-theme-analysis/outputs/topic_modeling/"$PREFIX"_all_input_docs.tsv --output /mnt/data0/lucy/literary-theme-analysis/outputs/topic_modeling/"$PREFIX"_input_text.mallet --keep-sequence

bash topic_modeling_helper.sh /mnt/data0/lucy/Mallet-202108/bin /mnt/data0/lucy/literary-theme-analysis/outputs/topic_modeling 100 "$PREFIX"