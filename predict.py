from model_base import ModelPredictionResults
from tensorflow_model import Code2VecModel
from config import Config


def load_model(config: Config) -> Code2VecModel:
    model = Code2VecModel(config)
    model.predict([])
    return model


def predict(model: Code2VecModel, ast_path: str) -> None:
    raw_result = model.predict([ast_path])
    if raw_result:
        print(parse_result_for_plugin(raw_result[0]))


def parse_result_for_plugin(raw_prediction: ModelPredictionResults) -> list:
    return list(map(list, zip(raw_prediction.topk_predicted_words, raw_prediction.topk_predicted_words_scores)))


if __name__ == "__main__":
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_model(config)

    while True:
        predict(model, input())
