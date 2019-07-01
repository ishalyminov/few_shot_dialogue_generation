import argparse
import copy
import json
import logging
import os

from utils.nlu_wrapper import NLUWrapper
from utils.task_list import add_task, execute_tasks, tasks

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

NLU = None


def initialize_nlu(in_host, in_port):
    global NLU
    NLU = NLUWrapper(in_host, in_port)


def process_dialogs_callback(in_params):
    dialogs, nlu, dataset_format, start_idx, end_idx, output_filename = in_params
    process_fn = process_dialogs_maluuba if dataset_format == 'maluuba' else process_dialogs_smd
    logger.info('Processing batch {} -- {}'.format(start_idx, end_idx))
    result = process_fn(dialogs[start_idx: end_idx], NLU)

    with open(output_filename, 'w') as dialogs_out:
        json.dump(result, dialogs_out)
    logger.info('Processed batch {} -- {}'.format(start_idx, end_idx))


def create_tasks(in_dialogs, in_nlu, in_dataset_format, in_result_folder, task_size=10):
    for i in range(0, len(in_dialogs), task_size):
        add_task((in_dialogs,
                  in_nlu,
                  in_dataset_format,
                  i,
                  min(i + task_size, len(in_dialogs)),
                  os.path.join(in_result_folder, '{}.json'.format(i))))


def process_dialogs_maluuba(in_dialogs, in_nlu):
    result = copy.deepcopy(in_dialogs)
    for dialog in result:
        for utterance in dialog['turns']:
            utterance['nlu'] = in_nlu.annotate(utterance['text'],
                                               modules=['Preprocessor', 'NEREnsemble', 'EntityLinker', 'NPDetector', 'SUTime'])
    return result


def process_dialogs_smd(in_dialogs, in_nlu):
    result = copy.deepcopy(in_dialogs)
    for dialog in result:
        for utterance in dialog['dialogue']:
            utterance['data']['nlu'] = in_nlu.annotate(utterance['data']['utterance'],
                                                       modules=['Preprocessor', 'NEREnsemble', 'EntityLinker', 'NPDetector', 'SUTime'])
    return result


def get_option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_folder')
    parser.add_argument('nlu_host')
    parser.add_argument('nlu_port')
    parser.add_argument('dataset_format', help='maluuba/smd')
    parser.add_argument('--jobs', type=int, default=8)
    return parser


if __name__ == '__main__':
    parser = get_option_parser()
    args = parser.parse_args()
    with open(args.input_file) as dataset_in:
        dataset = json.load(dataset_in)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    nlu = None # NLUWrapper(args.nlu_host, args.nlu_port)
    create_tasks(dataset, nlu, args.dataset_format, args.output_folder)

    logger.info('got {} tasks'.format(len(tasks)))

    if 1 < args.jobs:
        retcodes = execute_tasks(process_dialogs_callback, args.jobs, initializer=initialize_nlu, initargs=[args.nlu_host, args.nlu_port])
    else:
        initialize_nlu(args.nlu_host, args.nlu_port)
        retcodes = [process_dialogs_callback(task) for task in tasks]

