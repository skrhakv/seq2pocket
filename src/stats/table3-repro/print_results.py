#!/usr/bin/env python3
import pickle
import sys

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/stats/table3-repro')
import table3_core  # noqa: E402

RESULTS = {
    'gbs': f'{PROJECT_DIRECTORY}/src/stats/table3-repro/results-gbs.pkl',
    'cbs': f'{PROJECT_DIRECTORY}/src/stats/table3-repro/results-cbs.pkl',
}


def main():
    for task, path in RESULTS.items():
        with open(path, 'rb') as f:
            d = pickle.load(f)
        records, number_of_pockets = d['records'], d['number_of_pockets']
        print(f'{"=" * 20} {task.upper()} {"=" * 20}')
        print(f'Loaded {len(records)} pockets (denominator {number_of_pockets}) from {path}')
        table3_core.print_and_test(records, number_of_pockets)
        print()


if __name__ == '__main__':
    main()
