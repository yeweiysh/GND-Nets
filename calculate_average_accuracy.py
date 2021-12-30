import argparse
import numpy as np


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_name", help="input: set a path to the accuuracy file")

        args = parser.parse_args()

        test_accuracy = []

        with open(args.file_name, 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                t_acc = line[:-1]
                test_accuracy.append(float(t_acc))

        np_test_accuracy = np.array(test_accuracy)
        ave_test_accuracy = np.mean(np_test_accuracy[0:30])
        std_test_accuracy = np.std(np_test_accuracy[0:30])

        print('test accuracy / mean(std): {0:.5f}({1:.5f})'.format(ave_test_accuracy, std_test_accuracy))

    except IOError as e:
        print(e)
