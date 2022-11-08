import random
from model_base import ModelPredictionResults
from tensorflow_model import Code2VecModel
from config import Config


def load_model(conf: Config) -> Code2VecModel:
    code2vec_model = Code2VecModel(conf)
    code2vec_model.predict([])
    return code2vec_model


def predict(code2vec_model: Code2VecModel, ast_path: str) -> None:
    raw_result = code2vec_model.predict([ast_path])
    if raw_result:
        print(parse_result_for_plugin(raw_result[0]))


def parse_result_for_plugin(raw_prediction: ModelPredictionResults) -> list:
    return list(map(list, zip(map(parse_tokens_to_method_name, raw_prediction.topk_predicted_words),
                              raw_prediction.topk_predicted_words_scores)))


def parse_tokens_to_method_name(tokens: str) -> str:
    method_name = ""
    for i, token in enumerate(tokens.split("|")):
        if token and i:
            chars = list(token)
            chars[0] = chars[0].upper()
            token = "".join(chars)
        method_name += token
    return method_name


def pad_input(inp: str, conf: Config) -> str:
    return inp + " " * (conf.MAX_CONTEXTS - inp.count(" "))


def shorten_input(inp: str, conf: Config) -> str:
    split = inp.split(" ")
    random.shuffle(split)
    return " ".join(split[:min(conf.MAX_CONTEXTS + 1, len(split))])


if __name__ == "__main__":
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_model(config)

    while True:
        raw_input = shorten_input(pad_input(input(), config), config)
        predict(model, raw_input)
