#!/bin/bash 
set -e

PREFIX=describe_gpt-4o # name of your experiment, corresponding to passage_type in preprocess_text_for_lda.py

/path/to/your/Mallet-202108/bin/mallet import-file --input /path/to/your/"$PREFIX"_all_input_docs.tsv --output /path/to/your/"$PREFIX"_input_text.mallet --keep-sequence

bash topic_modeling_helper.sh /path/to/your/Mallet-202108/bin /path/to/your/outputs/topic_modeling 100 "$PREFIX"
