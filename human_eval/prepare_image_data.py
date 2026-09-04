import sys
import os
import os
import os.path as osp
import json
import argparse
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR))

from analysis_utils import prepare_data
from data_utils import build_single_item

RESTRICT_TO_VALID = True

REF_ANN_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/processed_annotations/consensus_majority_clean.json"
)  # preprocessed annotations for this project
COLORGRID_ANN_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/color_grid_data.json"
)  # original annotations (formatted)
GRID_IMG_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/label_studio_input/images"
)  # path to grid images
LS_DATA_PATH = osp.join(
    BASE_DIR, "color-grid-segment-annotations/data/label_studio_input/label_studio_input.json"
)  # label studio inputs

SAMPLE_INDEX_FILE = osp.join(
    BASE_DIR, "human_eval/sample_ids.json"
)
IMAGE_OUT_DIR = osp.join(
    BASE_DIR, "human_eval/images"
)


def main(args):

    # load data
    _, data_df, _, _ = prepare_data(REF_ANN_PATH, COLORGRID_ANN_PATH, LS_DATA_PATH)
    data_df = data_df.set_index("round_id")

    # load sample ids
    with open(args.sample_index_file, "r") as f:
        sample_ids = json.load(f)

    # create output directory if it doesn't exist
    os.makedirs(args.image_out_dir, exist_ok=True)

    image_data = dict()
    # generate images and fetch data for each sample id
    print(f"Generating images and data for {len(sample_ids)} sample IDs...")
    for sample_id in tqdm(sample_ids):
        entry = data_df.loc[sample_id]
        sample_images = list()
        # generate images for each grid in the entry
        for i, obj in enumerate(entry.objs):
            image_filename = f"{sample_id}:{i}.png"
            # make image
            image = build_single_item(obj)
            # save image and store data
            image.save(os.path.join(args.image_out_dir, image_filename))
            sample_images.append(image_filename)
        # store data for the sample id
        image_data[sample_id] = {
            "condition": entry.condition,
            "listener_order": entry.listener_order,
            "speaker_order": entry.speaker_order,
            "target": entry.target.item(),
            "images": sample_images,
        }

    # save image_data to json
    with open(os.path.join(args.image_out_dir, "image_data.json"), "w") as f:
        json.dump(image_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare image data for sampled rounds."
    )
    parser.add_argument(
        "--ref_ann_path",
        type=str,
        default=REF_ANN_PATH,
        help="Path to reference annotations JSON file.",
    )
    parser.add_argument(
        "--colorgrid_ann_path",
        type=str,
        default=COLORGRID_ANN_PATH,
        help="Path to color grid annotations JSON file.",
    )
    parser.add_argument(
        "--ls_data_path",
        type=str,
        default=LS_DATA_PATH,
        help="Path to label studio input JSON file.",
    )
    parser.add_argument(
        "--sample_index_file",
        type=str,
        default=SAMPLE_INDEX_FILE,
        help="Path to sample IDs JSON file.",
    )
    parser.add_argument(
        "--image_out_dir",
        type=str,
        default=IMAGE_OUT_DIR,
        help="Directory to save generated images.",
    )

    args = parser.parse_args()
    main(args)
