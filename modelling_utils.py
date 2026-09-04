from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import json
import os.path as osp
import base64

from data_utils import build_grid_image_with_target_highlight


def get_span_data(row):
    """
    Extract span data from a row of the aggregated DataFrame and return it as a list of dictionaries.
    """
    n_spans = len(row.span)
    all_spans = []

    if n_spans == 0:
        # if there are no spans
        output_format = {
            "annotations": [],
            "no_reference": True
        }
        return output_format

    for i in range(n_spans):
        span = row.span[i]
        span_text = row.span_text[i]
        label_set = row.label_set[i]

        span_annotation_data = {
            "span": span_text,
            "start": span[0],
            "end": span[1],
            "cells": order_label_set(label_set),
            "comment": ""
        }

        all_spans.append(span_annotation_data)

        output_format = {
            "annotations": all_spans,
            "no_reference": False
        }

    return output_format


def order_label_set(label_set): 
    """
    order an input label set consistently by grid and then by cell index
    example: ["d1_5", "d2_5", "d2_4", "t5", "t4"] -> ["t4", "t5", "d1_5", "d2_4", "d2_5"]
    """
    def label_key(label):
        if label.startswith("t"):
            return (0, int(label[1:]))  # targets come first
        elif label.startswith("d"):
            parts = label[1:].split("_")
            return (1, int(parts[0]), int(parts[1]))  # then distractor cells
        else:
            return (2, label)  # any other labels come last

    ordered_labels = sorted(label_set, key=label_key)
    return ordered_labels


def make_annotation_string(row):
    span_data = get_span_data(row)
    return json.dumps(span_data, indent=2)


def get_image_from_web(url):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content))
    return img


def get_image_from_local(filepath):
    img = Image.open(filepath)
    return img


def get_description(row):
    return row.full_text


def make_instruction_message(instruction_text):
    instruction_message = {
        "role": "system",
        "content": [{"type": "text", "text": instruction_text}],
    }

    return instruction_message


def make_input_message(data_url):

    sample_message = {
        "role": "user",
        "content": [
            {"type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ],
    }

    return sample_message


def make_data_url_from_image(image):
    """
        # Convert PIL image to base64 data url
    """

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"

    return data_url


def make_sample_message_for_openai(row, image_dir=None):
    description = get_description(row)
    if image_dir is not None:
        filename = osp.split(row.ann_url)[-1]
        filepath = osp.join(image_dir, filename)
        image = get_image_from_local(filepath)
    else:
        image = get_image_from_web(row.ann_url)

    # Convert PIL image to base64 data url
    data_url = make_data_url_from_image(image)
    
    sample_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": description},
            {"type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ],
    }

    return sample_message


def make_annotation_message(row):
    annotation_string = make_annotation_string(row)
    annotation_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": annotation_string}],
    }

    return annotation_message


def get_examples(
    possible_examples_df,
    n_examples=5,
    ensure_distractor_mention=False,
    ensure_multiturn=False,
    ensure_simple=False,
    ensure_invalid=False,
    invalid_description_text="speaker: A 3x3 grid of colored squares with a green border surrounding it."
):

    assert n_examples >= sum(
        [
            ensure_distractor_mention,
            ensure_multiturn,
            ensure_simple,
            ensure_invalid,
        ]
    ), "Number of examples must be at least equal to the number of enforced criteria."

    example_df = pd.DataFrame()

    if ensure_distractor_mention:
        # sample cases that include a reference to a distractor
        example = possible_examples_df.loc[
            possible_examples_df.includes_distractor_reference
        ].sample(1)
        example_df = pd.concat([example_df, example])
        possible_examples_df = possible_examples_df.drop(index=example.index)
        n_examples -= len(example)
    if ensure_multiturn:
        # sample cases that include multiple utterances and multiple spans
        example = possible_examples_df.loc[
            np.logical_and(
                possible_examples_df.n_utterances > 1, possible_examples_df.n_spans > 1
            )
        ].sample(1)
        example_df = pd.concat([example_df, example])
        possible_examples_df = possible_examples_df.drop(index=example.index)
        n_examples -= len(example)
    if ensure_simple:
        # sample cases that include only a single utterance and a single span, and do not include a reference to a distractor
        example = possible_examples_df.loc[
            np.logical_and(
                possible_examples_df.n_utterances == 1,
                possible_examples_df.n_spans == 1,
                possible_examples_df.includes_distractor_reference == False,
            )
        ].sample(1)
        example_df = pd.concat([example_df, example])
        possible_examples_df = possible_examples_df.drop(index=example.index)
        n_examples -= len(example)
    if ensure_invalid:
        # sample cases that include an artificial invalid reference
        example = possible_examples_df.sample(1)
        
        # manually overwrite description and span information of example
        example.full_text = invalid_description_text
        example.span = [[]]
        example.span_text = [[]]
        example.label_set = [[]]
        
        example_df = pd.concat([example_df, example])
        possible_examples_df = possible_examples_df.drop(index=example.index)
        n_examples -= len(example)

    # sample remaining examples randomly
    further_examples = possible_examples_df.sample(n_examples)
    example_df = pd.concat([example_df, further_examples])

    # shuffle order of examples
    example_df = example_df.sample(frac=1)

    return example_df


def make_sample_message_for_openai(row, image_dir=None, external_description=None):
    if isinstance(external_description, str):
        description = f"speaker: {external_description}"
    else:
        description = get_description(row)
    if image_dir is not None:
        filename = osp.split(row.ann_url)[-1]
        filepath = osp.join(image_dir, filename)
        image = get_image_from_local(filepath)
    else:
        image = get_image_from_web(row.ann_url)

    # Convert PIL image to base64 data url
    data_url = make_data_url_from_image(image)

    sample_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": description},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

    return sample_message


def make_input_messages(
    target_id,
    df,
    instruction_text,
    external_description=None,
    image_dir=None,
    n_examples=5,
    ensure_distractor_mention=True,
    ensure_multiturn=True,
    ensure_simple=True,
    ensure_invalid=True, 
    shuffle_examples=True
):

    ############################################
    # select target, identify remaining items
    ############################################

    target_row = df.loc[target_id]
    remaining_items = df[np.logical_not(df.index == target_id)]

    ############################################
    # build full prompt (with few-shot examples)
    ############################################

    instruction_message = make_instruction_message(instruction_text)

    # few-shot examples
    example_messages = []

    example_df = get_examples(
        remaining_items,
        n_examples,
        ensure_distractor_mention,
        ensure_multiturn,
        ensure_simple,
        ensure_invalid,
    )
    
    if shuffle_examples:
        example_df = example_df.sample(frac=1)

    for _, row in example_df.iterrows():
        # create sample message and annotation message for each example, and collect images
        sample_message = make_sample_message_for_openai(row, image_dir)
        annotation_message = make_annotation_message(row)
        example_messages += [sample_message, annotation_message]

    # query item: insert model description here if specified (external_description)
    query_message = make_sample_message_for_openai(
        target_row, image_dir, external_description
    )

    # full text input
    messages = [instruction_message, *example_messages, query_message]

    return messages


def make_model_input(
    row,
    instruction_text,
    patch_size=100,
    patch_padding=0,
    grid_padding=50,
    target_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_pad_color=(255, 255, 255),
    target_pad_color=(0, 255, 0),
):

    # build image and transform to b64 data url
    img = build_grid_image_with_target_highlight(
        row.objs,
        row.speaker_order,
        target_idx=row.target,
        patch_size=patch_size,
        patch_padding=patch_padding,
        grid_padding=grid_padding,
        target_padding=target_padding,
        patch_pad_color=patch_pad_color,
        grid_pad_color=grid_pad_color,
        target_pad_color=target_pad_color,
    )
    data_url = make_data_url_from_image(img)

    # instruction
    instruction_message = make_instruction_message(instruction_text)

    # query item
    input_message = make_input_message(data_url)

    # full text input
    messages = [instruction_message, input_message]

    return messages