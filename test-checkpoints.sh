for i in {1..14}
do
   echo "==================="
   echo "|| Checkpoint $i: ||"
   echo "==================="
   python code2vec.py --load models/java14m/saved_model_iter$i --test data/java-large/java-large.test.c2v > test-stats/checkpoint$i.txt
done
echo "==================="
echo "|| Saved Model    ||"
echo "==================="
python code2vec.py --load models-large/java14m/saved_model --test data/java-med/java-large.test.c2v > test-stats/saved_model.txt


