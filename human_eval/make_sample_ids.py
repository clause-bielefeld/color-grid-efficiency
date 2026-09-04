import sys
import os
from os import path as osp
import json
import argparse

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
BASE_DIR = osp.realpath(osp.join(SCRIPT_DIR, os.pardir))
sys.path.append(osp.join(BASE_DIR))

from analysis_utils import prepare_data

RESTRICT_TO_VALID = True

# default file paths (can be overwritten with command line arguments)
REF_ANN_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/processed_annotations/consensus_majority_clean.json"
)  # preprocessed annotations for this project
COLORGRID_ANN_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/color_grid_data.json"
)  # original annotations (formatted)
LS_DATA_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/label_studio_input/label_studio_input.json"
)  # label studio inputs


def get_utterances(utterances_list):
    assert len(utterances_list) == 1, "Expected a list with a single utterance"
    return utterances_list[0][1]  # return the text of the single utterance


def main(args):

    ann_df, data_df, _, _ = prepare_data(
        args.ref_ann_path, args.colorgrid_ann_path, args.ls_data_path
    )

    if RESTRICT_TO_VALID:
        print("restrict to valid annotations")
        ann_df = ann_df[ann_df.label_set.map(len) > 1]

    annotated_games = ann_df.game_id.unique()

    print("restrict to annotated games with single utterances")
    # restrict data_df to only annotated games
    _data_df = data_df[data_df.gameid.isin(annotated_games)]
    # restrict to only single utterance games
    _data_df = _data_df[_data_df.utterances.map(len) == 1]
    # restrict to successful rounds
    _data_df = _data_df[_data_df.success]

    print(f"sample {args.n_samples_per_condition} from each condition")
    # sample N_SAMPLES_PER_CONDITION from each condition
    sampled_df = _data_df.groupby("condition", group_keys=False).sample(
        n=args.n_samples_per_condition, random_state=args.random_state
    )

    print(f"save sampled ids to {osp.join(SCRIPT_DIR, 'sample_ids.json')}")
    sampled_idx = sampled_df.round_id.to_list()
    with open(osp.join(SCRIPT_DIR, "sample_ids.json"), "w") as f:
        json.dump(sampled_idx, f)

    print("make human sample")
    _sampled_df = sampled_df.copy().rename(columns={"round_id": "target_id"})
    _sampled_df["description"] = _sampled_df.utterances.map(get_utterances)
    _sampled_df["system"] = "human"

    print(f"save human sample to {osp.join(args.out_dir, 'sample_human.json')}")
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    _sampled_df[["target_id", "description", "system"]].reset_index(drop=True).to_json(
        osp.join(args.out_dir, "sample_human.json"), orient="records", indent=2
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sample IDs from each condition")
    # general arguments
    parser.add_argument(
        "--n_samples_per_condition",
        type=int,
        default=50,
        help="Number of samples per condition",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=123,
        help="Random state for reproducibility",
    )
    # data paths
    parser.add_argument(
        "--ref_ann_path",
        type=str,
        default=REF_ANN_PATH,
        help="Path to the reference annotations",
    )
    parser.add_argument(
        "--colorgrid_ann_path",
        type=str,
        default=COLORGRID_ANN_PATH,
        help="Path to the original color grid annotations",
    )
    parser.add_argument(
        "--ls_data_path",
        type=str,
        default=LS_DATA_PATH,
        help="Path to the label studio input data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=osp.join(SCRIPT_DIR, "samples"),
        help="Directory to save the human sample",
    )
    args = parser.parse_args()

    main(args)
