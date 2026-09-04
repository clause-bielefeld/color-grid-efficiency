import pandas as pd
from glob import glob
import os
from os import path as osp
import json
import argparse

MODEL_DESCRIPTION_DIR = osp.abspath(
    osp.join(os.pardir, "output", "generated_descriptions")
)
OUT_DIR = osp.abspath("samples")
SAMPLE_FILE = osp.abspath("sample_ids.json")


def parse_model_file(file):
    with open(file, "r") as f:
        data = json.load(f)

    model_id = data["model_id"]
    args = data["args"]
    responses_df = pd.DataFrame(data["responses"])

    return model_id, args, responses_df


def make_out_filename(model_id, args):

    sample_str = f"_n{args['limit']}" if args["limit"] is not None else ""
    thinking_str = "_thinking" if args["enable_thinking"] else ""

    out_filename = f"sample_{osp.split(model_id)[-1].replace('.', '').replace('_', '')}{thinking_str}{sample_str}.json"

    return out_filename


def process_model_file(file, sample_ids, out_dir, force_overwrite=False):

    model_id, args, responses_df = parse_model_file(file)

    out_filename = make_out_filename(model_id, args)
    out_path = osp.join(out_dir, out_filename)
    if osp.exists(out_path) and not force_overwrite:
        print(f"Sample file {out_path} already exists. Skipping...")
        return

    print(f"Processing model file {file} for model {model_id}...")
    sample_df = responses_df.loc[responses_df["target_id"].isin(sample_ids)]
    sample_df["system"] = model_id
    sample_df = sample_df[["target_id", "description", "system"]]

    sample_df.to_json(out_path, orient="records", indent=2)


def validate_samples(sample_files, sample_ids, verbose_errors=False):

    reference_ids = set(sample_ids)
    errors = []  # list to store model_ids with errors

    for file in sample_files:

        with open(file, "r") as f:
            sample_data = json.load(f)

        ids = {entry["target_id"] for entry in sample_data}
        descriptions = {entry["description"] for entry in sample_data}
        model_id = sample_data[0]["system"]

        # validate descriptions
        valid_descriptions = [
            isinstance(description, str) and len(description) > 0
            for description in descriptions
        ]
        all_valid_descriptions = all(valid_descriptions)
        
        # validate IDs
        all_valid_ids = ids == reference_ids
        
        # print validation results
        print(f"Validating sample file {file} for model {model_id}...")
        print(
            f"\t{'✔ Valid' if all_valid_descriptions else '❌ INVALID'} descriptions"
        )
        print(f"\t{'✔ Valid' if all_valid_ids else '❌ INVALID'} IDs")

        if not all_valid_descriptions or not all_valid_ids:
            errors.append(model_id)
            
            if verbose_errors: 
                if not all_valid_descriptions:
                    print(f"\tInvalid descriptions found for model {model_id}:")
                    for valid, description, target_id in zip(valid_descriptions, descriptions, ids):
                        if not valid:
                            print("\t" * 2 + f"target_id: {target_id}, description: {description}")                
                if not all_valid_ids:
                    print(f"\tInvalid IDs found for model {model_id}:")
                    missing_ids = reference_ids - ids
                    extra_ids = ids - reference_ids
                    if missing_ids:
                        print("\t" * 2 + f"Missing IDs: {missing_ids}")
                    if extra_ids:
                        print("\t" * 2 + f"Extra IDs: {extra_ids}")

    if len(errors) == 0:
        print("\n\nNo errors found in sample files.")
    else:
        print(f"\n\nErrors found in {len(errors)} model files: {errors}")


def main(args):

    # assert that the sample file exists
    assert osp.isfile(
        args.sample_file
    ), f"Sample file {args.sample_file} does not exist, run make_sample_ids.py first."

    # load sample IDs
    with open(args.sample_file, "r") as f:
        sample_ids = json.load(f)
    print(f"\n{len(sample_ids)} sample IDs loaded from {args.sample_file}.")

    if not args.validate_only:

        print("\nGenerating sample files from model description files...")
        
        # retrieve model files
        model_files = glob(osp.join(args.model_description_dir, "*.json"))
        print(f"{len(model_files)} model files found in {args.model_description_dir}.")

        # create output directo ry if it doesn't exist
        os.makedirs(args.out_dir, exist_ok=True)

        # process each model file and create sample files
        for file in model_files:
            process_model_file(
                file, sample_ids, args.out_dir, force_overwrite=args.force_overwrite
            )

    else:
        print("\nValidation only mode enabled. Skipping sample generation.")

    # validate the generated sample files
    print("\nValidating generated sample files...")
    sample_files = glob(osp.join(args.out_dir, "*.json"))
    validate_samples(sample_files, sample_ids, verbose_errors=args.verbose_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make samples from model description files."
    )
    parser.add_argument(
        "--model_description_dir",
        type=str,
        default=MODEL_DESCRIPTION_DIR,
        help="Directory containing model description JSON files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help="Directory to save the sample JSON files.",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default=SAMPLE_FILE,
        help="JSON file containing the list of sample IDs.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="If set, will overwrite existing sample files.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="If set, will only validate existing sample files without generating new ones.",
    )
    parser.add_argument(
        "--verbose_errors",
        action="store_true",
        help="If set, will print detailed error messages during validation.",
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    main(args)
