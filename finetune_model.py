import time
from functools import partial
from typing import Optional

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from common import common
from config import Config
from model_base import ModelEvaluationResults
from path_context_reader import PathContextReader, EstimatorAction
from tensorflow_model import Code2VecModel, _TFTrainModelInputTensorsFormer, _TFEvaluateModelInputTensorsFormer, \
    SubtokensEvaluationMetric, TopKAccuracyEvaluationMetric
import tensorflow as tf
import numpy as np

from vocabularies import Code2VecVocabs, VocabType


class FinetuneModel(Code2VecModel):
    def __init__(self, pretrained_model: Code2VecModel, config, *args, **kwargs):
        super().__init__(config)
        self.pretrained_model = pretrained_model
        self.sess = tf.compat.v1.Session()
        self.log = config.get_logger().info
        self.config = config
        self.vocabs = Code2VecVocabs(config)

    def _build_tf_training_graph(self, input_tensors, trainable=False):
        # Use `_TFTrainModelInputTensorsFormer` to access input tensors by name.
        input_tensors = _TFTrainModelInputTensorsFormer().from_model_input_form(input_tensors)
        # shape of (batch, 1) for input_tensors.target_index
        # shape of (batch, max_contexts) for others:
        #   input_tensors.path_source_token_indices, input_tensors.path_indices,
        #   input_tensors.path_target_token_indices, input_tensors.context_valid_mask

        with tf.compat.v1.variable_scope('model',reuse=True):
            tokens_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Token],
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"),
                trainable=trainable)

            attention_param = tf.compat.v1.get_variable(
                'ATTENTION',
                shape=(self.config.CODE_VECTOR_SIZE, 1), dtype=tf.float32, trainable=trainable)
            paths_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Path],
                shape=(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"),
                trainable=trainable)

            code_vectors, _ = self._calculate_weighted_contexts(
                tokens_vocab, paths_vocab, attention_param, input_tensors.path_source_token_indices,
                input_tensors.path_indices, input_tensors.path_target_token_indices, input_tensors.context_valid_mask,
                trainable=trainable)

        with tf.compat.v1.variable_scope('model2'):
            targets_vocab = tf.compat.v1.get_variable(
                "TARGETS",
                shape=(self.config.CODE_VECTOR_SIZE, 3*384), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            targets_vocab2 = tf.compat.v1.get_variable(
                "TARGETS2",
                shape=(3*384, self.vocabs.target_vocab.size), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            intermediate = tf.sigmoid(tf.matmul(code_vectors, targets_vocab))
            logits = tf.sigmoid(tf.matmul(intermediate, targets_vocab2))
            batch_size = tf.cast(tf.shape(input_tensors.target_index)[0], dtype=tf.float32)

            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(input_tensors.target_index, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def _build_tf_test_graph(self, input_tensors, normalize_scores=False):
        # Use `_TFTrainModelInputTensorsFormer` to access input tensors by name.
        trainable = False
        input_tensors = _TFEvaluateModelInputTensorsFormer().from_model_input_form(input_tensors)
        # shape of (batch, 1) for input_tensors.target_index
        # shape of (batch, max_contexts) for others:
        #   input_tensors.path_source_token_indices, input_tensors.path_indices,
        #   input_tensors.path_target_token_indices, input_tensors.context_valid_mask

        with tf.compat.v1.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Token],
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"),
                trainable=trainable)

            attention_param = tf.compat.v1.get_variable(
                'ATTENTION',
                shape=(self.config.CODE_VECTOR_SIZE, 1), dtype=tf.float32, trainable=trainable)
            paths_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Path],
                shape=(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"),
                trainable=trainable)

            code_vectors, _ = self._calculate_weighted_contexts(
                tokens_vocab, paths_vocab, attention_param, input_tensors.path_source_token_indices,
                input_tensors.path_indices, input_tensors.path_target_token_indices, input_tensors.context_valid_mask,
                trainable=trainable)

        with tf.compat.v1.variable_scope('model2', reuse=self.get_should_reuse_variables()):
            targets_vocab = tf.compat.v1.get_variable(
                "TARGETS",
                shape=(self.config.CODE_VECTOR_SIZE, 3 * 384), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            targets_vocab2 = tf.compat.v1.get_variable(
                "TARGETS2",
                shape=(3 * 384, self.vocabs.target_vocab.size), dtype=tf.float32,
                initializer=tf.compat.v1.initializers.variance_scaling(scale=1.0, mode='fan_out',
                                                                       distribution="uniform"))
            intermediate = tf.sigmoid(tf.matmul(code_vectors, targets_vocab))

            # Use `_TFEvaluateModelInputTensorsFormer` to access input tensors by name.
            # input_tensors = _TFEvaluateModelInputTensorsFormer().from_model_input_form(input_tensors)
            # shape of (batch, 1) for input_tensors.target_string
            # shape of (batch, max_contexts) for the other tensors

        scores = tf.matmul(intermediate, targets_vocab2)

        topk_candidates = tf.nn.top_k(scores, k=tf.minimum(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION, self.vocabs.target_vocab.size))
        top_indices = topk_candidates.indices
        top_words = self.vocabs.target_vocab.lookup_word(top_indices)
        original_words = input_tensors.target_string
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, _, input_tensors.path_source_token_strings, \
            input_tensors.path_strings, input_tensors.path_target_token_strings, code_vectors


    def evaluate(self) -> Optional[ModelEvaluationResults]:
        self._initialize_session_variables()
        return super().evaluate()

    def _load_pretrained_model(self, sess=None):
        pass

if __name__ == "__main__":
    config = Config(set_defaults=True, load_from_args=True, verify=True)
    pretrained_model = Code2VecModel(config)
    pretrained_model.initialize_finetuning()
    fine_model = FinetuneModel(pretrained_model, config)
    fine_model.train()
    # pretrained_model.training_status = False
    # model = FinetuneModel(pretrained_model)
