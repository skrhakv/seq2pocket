#!/usr/bin/env python3

import argparse
import sys

sys.path.append('/home/skrhakv/Projects/seq2pocket/src/stats/table3-repro')
import table3_core

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'

MODEL_PATH = f'{PROJECT_DIRECTORY}/data/models/gbs-model-enhanced-scPDB-filtered.pt'
ESM_MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
ESM_EMBEDDINGS_PATH = f'{PROJECT_DIRECTORY}/data/embeddings/ligysis'
COORDINATES_DIR = f'{PROJECT_DIRECTORY}/data/coordinates/ligysis'
LIGYSIS_ANNOTATION_PATH = f'{PROJECT_DIRECTORY}/data/data-extraction/ligysis_for_pocket_level_evaluation.csv'
OUTPUT_PATH = f'{PROJECT_DIRECTORY}/src/stats/table3-repro/results-gbs.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N proteins (for a quick test run).')
    args = parser.parse_args()

    table3_core.run_evaluation(
        model_path=MODEL_PATH,
        esm_model_name=ESM_MODEL_NAME,
        embeddings_dir=ESM_EMBEDDINGS_PATH,
        coordinates_dir=COORDINATES_DIR,
        annotation_path=LIGYSIS_ANNOTATION_PATH,
        pocket_types=['NON_CRYPTIC'],
        output_path=OUTPUT_PATH,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
