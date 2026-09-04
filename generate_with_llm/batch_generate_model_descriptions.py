import sys
import os
from os import path as osp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import json
import argparse
import asyncio

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
BASE_DIR = osp.realpath(osp.join(SCRIPT_DIR, os.pardir, os.pardir))
sys.path.append(osp.join(BASE_DIR, "modelling"))

from analysis_utils import prepare_data
from modelling_utils import make_model_input

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
INSTRUCTION_FILE = osp.join(
    BASE_DIR, "modelling/model_instructions/model_description_guidelines_v2.txt"
)


def truncate_dict(d, max_len=200):
    return {k: (str(v)[:max_len] + "[...]" if isinstance(v, str) and len(v) > max_len else v) for k, v in d.items()}


async def main(args):

    with open(args.instruction_file, "r") as f:
        instruction_text = f.read()


    print("preparing data...")

    ann_df, data_df, _, _ = prepare_data(
        args.ref_ann_path, args.colorgrid_ann_path, args.ls_data_path
    )

    data_subset = data_df.loc[data_df.gameid.isin(ann_df.game_id)].set_index("round_id")

    if args.limit: 
        data_subset = data_subset.iloc[:args.limit]

    #
    # SETUP CLIENT
    #

    print("setup client...")

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.llm_server_ip}:{args.port}/v1",
    )

    models = await client.models.list()
    model_id = models.data[0].id

    print(f"set up client with model {model_id}")

    #
    # GENERATE ANNOTATIONS
    #


    async def single_request(
            target_id, target_row, semaphore, 
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, 
            presence_penalty=args.presence_penalty, enable_thinking=args.enable_thinking):
                
        input_messages = make_model_input(
            target_row,
            instruction_text,
            patch_size=args.patch_size,
            patch_padding=args.patch_padding,
            grid_padding=args.grid_padding,
            target_padding=args.target_padding
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
                    }
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
                    "target_id": target_id, "description": content, "reasoning": reasoning,
                    "error_type": None, "error_msg": None, "status_code": None
                }
            except Exception as e:
                # catch any exceptions and include the error message in the output
                out_obj = {"target_id": target_id, "description": None, "reasoning": None,
                           "error_type": type(e).__name__, "error_msg": str(e), "status_code": getattr(e, "status_code", None)
                }

            return out_obj


    async def batch_requests(data, max_concurrent):
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            single_request(target_id, target_row, semaphore)
            for target_id, target_row in data.iterrows()
        ]
        return await tqdm_asyncio.gather(*tasks)


    # generate and collect responses
    all_responses = await batch_requests(data_subset, max_concurrent=args.max_concurrent)

    #
    # SAVE GENERATED ANNOTATIONS
    #

    sample_str = (
        f"_n{args.limit}"
        if args.limit is not None
        else ""
    )
    thinking_str = "_thinking" if args.enable_thinking else ""

    out_path = osp.join(
        args.out_dir,
        f"{osp.split(model_id)[-1].replace('.', '').replace('_', '')}{thinking_str}{sample_str}_descriptions.json",
        )

    out_obj = {"model_id": model_id, "args": {**vars(args)}, "responses": all_responses}

    print(f"saving generated annotations to {out_path}...")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, out_path), "w") as f:
        json.dump(out_obj, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model descriptions for the color grid dataset."
    )
    
    ######################
    # general setup args #
    ######################

    parser.add_argument(
        "--out_dir",
        type=str,
        default=osp.join(SCRIPT_DIR, "output", "generated_descriptions"),
        help="Directory to save the generated descriptions.",
    )
    parser.add_argument("--limit", type=int, default=None)

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
        "--ls_data_path",
        type=str,
        default=LS_DATA_PATH,
        help="path to label studio inputs",
    )
    parser.add_argument("--instruction_file", type=str, default=INSTRUCTION_FILE)
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
    
    ###################
    # generation args #
    ###################
    
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
        help="Maximum number of concurrent requests to the model"
    )

    #########################
    # image generation args #
    #########################

    parser.add_argument(
        "--patch_size",
        type=int,
        default=100,
        help="Size of the individual patches in the generated images",
    )
    parser.add_argument(
        "--patch_padding",
        type=int,
        default=0,
        help="Padding between cells in the generated images",
    )
    parser.add_argument(
        "--grid_padding",
        type=int,
        default=50,
        help="Padding between grids in the generated images",
    )
    parser.add_argument(
        "--target_padding",
        type=int,
        default=50,
        help="Padding around the target in the generated images",
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    asyncio.run(main(args))
