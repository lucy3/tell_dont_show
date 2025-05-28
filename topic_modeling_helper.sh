BIN_DIR=$1
DIRECTORY=$2
TOPICS=$3
PREFIX=$4
echo "Data directory is" $DIRECTORY
echo "Number of topics is" $TOPICS
echo "Prefix is" $PREFIX

$BIN_DIR/mallet train-topics --input $DIRECTORY/"$PREFIX"_input_text.mallet \
      --num-topics $TOPICS \
      --output-state $DIRECTORY/"$PREFIX"_"$TOPICS"_topic-state.gz \
      --output-model $DIRECTORY/"$PREFIX"_"$TOPICS"_topic-model \
      --output-doc-topics $DIRECTORY/"$PREFIX"_"$TOPICS"_doc-topics \
      --output-topic-keys $DIRECTORY/"$PREFIX"_"$TOPICS"_topic-words \
      --inferencer-filename $DIRECTORY/"$PREFIX"_"$TOPICS"_inferencer