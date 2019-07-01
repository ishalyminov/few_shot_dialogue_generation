import json
import os
import sys


def main(in_folder, in_result_filename):
    result = []
    for filename in sorted(os.listdir(in_folder), key=lambda x: int(os.path.splitext(x)[0])):
        with open(os.path.join(in_folder, filename)) as json_in:
            result += json.load(json_in)
    with open(in_result_filename, 'w') as json_out:
        json.dump(result, json_out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
