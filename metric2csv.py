import argparse
import logging
import csv




def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    return parser


def _main(metric_path: str, output_path: str) -> None:
    list_metrics = []
    keys = ['checkpoint', 'precision', 'recall', 'F1']
    with open(metric_path) as metric_file:
        lines = metric_file.readlines()
        for line in lines:
            split_content = line.replace(":","").replace(",","").split()
            values = split_content[1::2]
            dictionary = {}
            for (key, value) in zip(keys, values):
                dictionary[key] = value
            list_metrics.append(dictionary)
    with open(output_path, 'w',  newline='\n') as output_file:
        fc = csv.DictWriter(output_file, fieldnames=keys,  delimiter=',')
        fc.writeheader()
        fc.writerows(list_metrics)




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    args = _build_argument_parser().parse_args()
    metric_path = args.metric_file
    output_path = args.output_file
    _main(metric_path, output_path)
