import time

from keras import Model

from config import Config
from path_context_reader import PathContextReader, EstimatorAction
from tensorflow_model import Code2VecModel, _TFTrainModelInputTensorsFormer
import tensorflow as tf
import numpy as np

from vocabularies import Code2VecVocabs


class FinetuneModel(Model):
    def __init__(self, pretrained_model: Code2VecModel, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = pretrained_model
        self.sess = tf.compat.v1.Session()
        self.log = config.get_logger().info
        self.config = config
        self.vocabs = Code2VecVocabs(config)

    def train(self):
        self.log('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_save_and_eval = max(int(self.config.train_steps_per_epoch * self.config.SAVE_EVERY_EPOCHS), 1)

        train_reader = PathContextReader(vocabs=self.vocabs,
                                         model_input_tensors_former=_TFTrainModelInputTensorsFormer(),
                                         config=self.config, estimator_action=EstimatorAction.Train)
        input_iterator = tf.compat.v1.data.make_initializable_iterator(train_reader.get_dataset())
        input_iterator_reset_op = input_iterator.initializer
        input_tensors = input_iterator.get_next()

        input_tens, code_vectors = self.pretrained_model._build_tf_fine_tune_training_graph(input_tensors, trainable=False)
        optimizer, train_loss = self._build_tf_training_graph(input_tens, code_vectors)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

        self.log('Number of trainable params: {}'.format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))
        for variable in tf.compat.v1.trainable_variables():
            self.log("variable name: {} -- shape: {} -- #params: {}".format(
                variable.name, variable.get_shape(), np.prod(variable.get_shape().as_list())))

        self._initialize_session_variables()

        if self.config.MODEL_LOAD_PATH:
            self.pretrained_model._load_inner_model()

        self.sess.run(input_iterator_reset_op)
        time.sleep(1)
        self.log('Started reader...')
        try:
            while True:
                # Each iteration = batch. We iterate as long as the tf iterator (reader) yields batches.
                batch_num += 1

                # Actual training for the current batch.
                _, batch_loss = self.sess.run([optimizer, train_loss])
                self.sess.run(input_iterator_reset_op)

                sum_loss += batch_loss
                if batch_num % self.config.NUM_BATCHES_TO_LOG_PROGRESS == 0:
                    self._trace_training(sum_loss, batch_num, multi_batch_start_time)
                    # Uri: the "shuffle_batch/random_shuffle_queue_Size:0" op does not exist since the migration to the new reader.
                    # self.log('Number of waiting examples in queue: %d' % self.sess.run(
                    #    "shuffle_batch/random_shuffle_queue_Size:0"))
                    sum_loss = 0
                    multi_batch_start_time = time.time()
                if batch_num % num_batches_to_save_and_eval == 0:
                    epoch_num = int((batch_num / num_batches_to_save_and_eval) * self.config.SAVE_EVERY_EPOCHS)
                    model_save_path = self.config.MODEL_SAVE_PATH + '_iter' + str(epoch_num)
                    self.save(model_save_path)
                    self.log('Saved after %d epochs in: %s' % (epoch_num, model_save_path))
                    evaluation_results = self.evaluate()
                    evaluation_results_str = (str(evaluation_results).replace('topk', 'top{}'.format(
                        self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)))
                    self.log('After {nr_epochs} epochs -- {evaluation_results}'.format(
                        nr_epochs=epoch_num,
                        evaluation_results=evaluation_results_str
                    ))
                    if epoch_num == self.config.NUM_TRAIN_EPOCHS:
                        break
                print(batch_loss)

        except tf.errors.OutOfRangeError as err:
            print(err)
            print('End of dataset')

    def _build_tf_training_graph(self, input_tensors, code_vectors):
        # Use `_TFTrainModelInputTensorsFormer` to access input tensors by name.
        # shape of (batch, 1) for input_tensors.target_index
        # shape of (batch, max_contexts) for others:
        #   input_tensors.path_source_token_indices, input_tensors.path_indices,
        #   input_tensors.path_target_token_indices, input_tensors.context_valid_mask

        with tf.compat.v1.variable_scope('model2'):
            targets_vocab = tf.compat.v1.get_variable(
                "TARGETS",
                shape=(self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            targets_vocab2 = tf.compat.v1.get_variable(
                "TARGETS2",
                shape=(self.config.TARGET_EMBEDDINGS_SIZE, self.config.TARGET_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            intermediate = tf.matmul(targets_vocab, targets_vocab2)
            logits = tf.matmul(code_vectors, intermediate, transpose_b=True)
            batch_size = tf.cast(tf.shape(input_tensors.target_index)[0], dtype=tf.float32)

            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(input_tensors.target_index, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def _initialize_session_variables(self):
        self.sess.run(tf.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer(),
            tf.compat.v1.tables_initializer()))
        self.log('Initalized variables')


if __name__ == "__main__":
    config = Config(set_defaults=True, load_from_args=True, verify=True)
    pretrained_model = Code2VecModel(config)
    fine_model = FinetuneModel(pretrained_model, config)
    fine_model.train()
    # pretrained_model.training_status = False
    # model = FinetuneModel(pretrained_model)
