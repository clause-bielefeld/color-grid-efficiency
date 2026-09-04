import sys
import os
import os.path as osp
from itertools import chain
import json
import argparse
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
BASE_DIR = osp.realpath(osp.join(SCRIPT_DIR, os.pardir, os.pardir))
sys.path.append(osp.join(BASE_DIR, "modelling"))

from modelling_utils import make_input_messages
from analysis_utils import prepare_data

# default file paths (can be overwritten with command line arguments)
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
INSTRUCTION_FILE = osp.join(
    BASE_DIR, "modelling/model_instructions/model_annotation_guidelines_v3.txt"
)



async def main(args):

    with open(args.instruction_file, "r") as f:
        INSTRUCTION_TEXT = f.read()

    print("preparing data...")

    ann_df, data_df, _, _ = prepare_data(
        args.ref_ann_path, args.colorgrid_ann_path, args.ls_data_path
    )

    if not args.not_restrict_to_valid:
        print("restrict to valid annotations")
        ann_df = ann_df[ann_df.label_set.map(len) > 1]

    #
    # DATA PREPARATION
    #

    SELECTED_COLUMNS = [
        "game_id",
        "round_num",
        "round_id",
        "start",
        "end",
        "span_text",
        "full_text",
        "label_set",
        "ann_url",
        "condition",
    ]

    ann_df_reduced = ann_df[SELECTED_COLUMNS]
    ann_df_reduced["span"] = ann_df_reduced.apply(lambda x: (x.start, x.end), axis=1)
    ann_df_reduced = ann_df_reduced.drop(columns=["start", "end"])
    ann_df_reduced.label_set = ann_df_reduced.label_set.map(
        lambda label_set: [x for x in label_set if x != "with_reference"]
    )
    ann_df_reduced["n_utterances"] = ann_df.round_id.map(
        data_df.set_index("round_id").n_utterances
    )

    first_aggregate = [
        "game_id",
        "round_num",
        "round_id",
        "full_text",
        "ann_url",
        "n_utterances",
        "condition",
    ]
    collapsed_df = (
        ann_df_reduced.groupby("round_id")
        .agg(
            {
                c: ("first" if c in first_aggregate else list)
                for c in ann_df_reduced.columns
            }
        )
        .sort_values(by=["game_id", "round_num"])
    )

    collapsed_df["unique_labels"] = collapsed_df.label_set.apply(
        lambda x: set(chain(*x))
    )
    collapsed_df["n_unique_labels"] = collapsed_df.unique_labels.map(len)
    collapsed_df["includes_distractor_reference"] = collapsed_df.unique_labels.map(
        lambda x: any([label.startswith("d") for label in x])
    )
    collapsed_df["n_spans"] = collapsed_df.label_set.map(len)

    #
    # SETUP CLIENT
    #

    print("setup client...")

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.llm_server_ip}:{args.port}/v1",
        timeout=args.timeout,
        max_retries=args.max_retries
    )

    models = await client.models.list()
    model_id = models.data[0].id

    print(f"set up client with model {model_id}")

    #
    # GENERATE ANNOTATIONS
    #

    if not isinstance(args.sample_idx_file, str):
        print("using all data for evaluation")
        eval_dataset = collapsed_df.copy()
    else:
        print(f"loading sample indices from {args.sample_idx_file}...")
        with open(args.sample_idx_file, "r") as f:
            sample_idx = json.load(f)
        eval_dataset = collapsed_df.loc[sample_idx]

    async def single_request(
        target_id,
        semaphore,
        df=collapsed_df,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        enable_thinking=args.enable_thinking,
        image_dir=args.grid_img_path,
        n_examples=args.n_examples,
    ):

        input_messages = make_input_messages(
            target_id,
            df,
            INSTRUCTION_TEXT,
            external_description=None,
            image_dir=image_dir,
            n_examples=n_examples,
            ensure_distractor_mention=True,
            ensure_multiturn=True,
            ensure_simple=True,
            ensure_invalid=True, 
            shuffle_examples=True
        )

        async with semaphore:
            try:
                chat_response = await client.chat.completions.create(
                    model=model_id,
                    messages=input_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )

                response_dict = dict(chat_response.choices[0].message)

                content = response_dict.get("content")
                if "reasoning" in response_dict.keys():
                    # vLLM
                    reasoning = response_dict.get("reasoning")
                elif "reasoning_content" in response_dict.keys():
                    # llama.cpp
                    reasoning = response_dict.get("reasoning_content")
                else:
                    # fallback / non-reasoning models
                    reasoning = None

                out_obj = {
                    "target_id": target_id,
                    "model_response": content,
                    "reasoning": reasoning,
                    "error_type": None,
                    "error_msg": None,
                    "status_code": None,
                }
            except Exception as e:
                # catch any exceptions and include the error message in the output
                out_obj = {
                    "target_id": target_id,
                    "model_response": None,
                    "reasoning": None,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "status_code": getattr(e, "status_code", None),
                }

            return out_obj

    async def batch_requests(data, max_concurrent=args.max_concurrent):
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            single_request(target_id, semaphore) for target_id, _ in data.iterrows()
        ]
        return await tqdm_asyncio.gather(*tasks)

    # generate and collect responses
    all_responses = await batch_requests(eval_dataset)

    #
    # SAVE GENERATED ANNOTATIONS
    #

    if args.sample_idx_file is not None:
        sample_str = "_" + osp.splitext(osp.basename(args.sample_idx_file))[0]
    else:
        sample_str = "_full"
        
    thinking_str = "-thinking" if args.enable_thinking else ""
    eval_system_str = "_human"
        
    out_path = osp.join(
        args.out_dir,
        f"{osp.split(model_id)[-1].replace('.', '').replace('_', '')}{thinking_str}{sample_str}{eval_system_str}_annotations.json",
        )

    out_obj = {"model_id": model_id, "evaluated_system": "human", "args": {**vars(args)}, "responses": all_responses}

    print(f"saving generated annotations to {out_path}...")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, out_path), "w") as f:
        json.dump(out_obj, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model annotations for the color grid dataset."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=osp.join(SCRIPT_DIR, "output", "generated_annotations"),
        help="Directory to save the generated annotations.",
    )
    parser.add_argument("--sample_idx_file", type=str, default=None)

    parser.add_argument(
        "--ref_ann_path",
        type=str,
        default=REF_ANN_PATH,
        help="path to preprocessed annotations for this project",
    )
    parser.add_argument(
        "--colorgrid_ann_path",
        type=str,
        default=COLORGRID_ANN_PATH,
        help="path to original annotations (formatted)",
    )
    parser.add_argument(
        "--grid_img_path", type=str, default=GRID_IMG_PATH, help="path to grid images"
    )
    parser.add_argument(
        "--ls_data_path",
        type=str,
        default=LS_DATA_PATH,
        help="path to label studio inputs",
    )
    parser.add_argument("--instruction_file", type=str, default=INSTRUCTION_FILE)
    parser.add_argument(
        "--not_restrict_to_valid",
        action="store_true",
        help="if set, do not restrict to valid annotations",
    )

    parser.add_argument(
        "--llm_server_ip",
        type=str,
        default="localhost",
        help="IP address of the vLLM API server",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Port of the vLLM API server",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for requests to the vLLM API server (in seconds)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for failed requests to the vLLM API server",
    )

    ###################
    # generation args #
    ###################

    parser.add_argument(
        "--n_examples",
        type=int,
        default=5,
        help="Number of examples to include in the prompt for each target",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the generation process",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter for the generation process",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty for the generation process",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to enable the model's 'thinking' process (reasoning trace) during generation",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=20,
        help="Maximum number of concurrent requests to the model",
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    asyncio.run(main(args))
