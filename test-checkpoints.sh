DATASET_NAME=$1
MODEL_NAME=$2
TAG=$MODEL_NAME/
EVALUATION_SET=$3
START=$4
END=$5

TEST_STATS=test-stats
SAVE_DIR=$TEST_STATS/$DATASET_NAME/$TAG
MODEL=models/$DATASET_NAME/$MODEL_NAME
TEST_SET=data/$DATASET_NAME/$DATASET_NAME.$EVALUATION_SET.c2v
EXPORT_FILE_PATH=$SAVE_DIR/metrics.txt
EXPORT_CSV_FILE_PATH=$MODEL/metrics-$EVALUATION_SET.csv

mkdir -p "$SAVE_DIR"
for i in $(eval echo "{$START..$END}")
do
   echo "==================="
   echo "|| Checkpoint $i: ||"
   echo "==================="
   CHECKPOINT=$SAVE_DIR/checkpoint"$i".txt
   python code2vec.py --load "$MODEL"/saved_model_iter"$i" --test "$TEST_SET"  > "$CHECKPOINT"  2>/dev/null
    searchstring='precision'
   t=$(ls -l | grep $searchstring "$CHECKPOINT")
   prefix=${t%%$searchstring*}
   echo "checkpoint $i: ${t:(${#prefix})}" >> "$EXPORT_FILE_PATH"
   echo "checkpoint $i: ${t:(${#prefix})}"
done

echo "Converting to csv..."
rm "$SAVE_DIR"/checkpoint*.txt
python metric2csv.py --metric-file="$EXPORT_FILE_PATH" --output-file="$EXPORT_CSV_FILE_PATH"
rm -rf "$SAVE_DIR"
rm "log.txt"
echo "Finished!"
