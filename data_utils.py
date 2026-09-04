from os import path as osp
import numpy as np
import colorsys
from PIL import Image
from PIL import ImageOps
import json
from itertools import chain
import re

# building item images


def hsl_to_rgb(h, s, l):
    h = h / 360 if h > 1 else h
    s = s / 100 if s > 1 else s
    l = l / 100 if l > 1 else l
    return colorsys.hls_to_rgb(h, l, s)


def hls_to_rgb(h, l, s):
    h = h / 360 if h > 1 else h
    l = l / 100 if l > 1 else l
    s = s / 100 if s > 1 else s
    return colorsys.hls_to_rgb(h, l, s)


def pil_format(rgb_tuple):
    rgb_array = np.array(rgb_tuple)
    rgb_array = (rgb_array * 255).astype(np.uint8).reshape(1, 1, -1)
    return rgb_array


def pad_img(img, size, color):
    return ImageOps.pad(img, size, color=color)


def surround_with_padding(img, padding=10, color=(255, 255, 255)):
    size = max(img.size)
    new_size = int(size + padding)
    out_img = pad_img(
        pad_img(img, (new_size, size), color), (new_size, new_size), color
    )
    return out_img


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def build_grid_image(
    objs,
    order,
    patch_size=100,
    patch_padding=10,
    grid_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_pad_color=(255, 255, 255),
):

    obj_grids = []

    for idx in order:
        # select next grid (following listener order)
        obj = objs[idx]
        # make grid
        padded_grid = build_single_item(
            obj,
            patch_size=patch_size,
            patch_padding=patch_padding,
            patch_pad_color=patch_pad_color,
            grid_padding=grid_padding,
            grid_pad_color=grid_pad_color,
        )
        obj_grids.append(padded_grid)

    # concatenate grids
    item_img = image_grid(obj_grids, 1, 3)

    return item_img


def build_grid_image_with_target_highlight(
    objs,
    order,
    patch_size=100,
    patch_padding=10,
    grid_padding=50,
    target_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_pad_color=(255, 255, 255),
    target_idx=None,
    target_pad_color=(0, 255, 0),
):

    obj_grids = []

    for idx in order:
        # select next grid (following listener order)
        obj = objs[idx]
        # make grid
        grid = build_single_item(
            obj=obj,
            patch_size=patch_size,
            patch_padding=patch_padding,
            patch_pad_color=patch_pad_color,
            grid_padding=0,
        )
        if target_idx is not None and idx == target_idx:
            # highlight target
            grid = surround_with_padding(grid, round(target_padding), (patch_pad_color))
            grid = surround_with_padding(grid, round(target_padding), (target_pad_color))
        else:
            # default pad color
            grid = surround_with_padding(grid, round(target_padding), (patch_pad_color))
            grid = surround_with_padding(grid, round(target_padding), (patch_pad_color))

        # pad grid
        padded_grid = surround_with_padding(grid, grid_padding, color=grid_pad_color)

        obj_grids.append(padded_grid)

    # concatenate grids
    item_img = image_grid(obj_grids, 1, 3)

    return item_img


def build_single_item(
    obj,
    patch_size=100,
    patch_padding=10,
    patch_pad_color=(255, 255, 255),
    grid_padding=50,
    grid_pad_color=(255, 255, 255),
):

    # patches to rgb
    rgbs = [pil_format(hsl_to_rgb(*shape["color"])) for shape in obj["shapes"]]
    # PIL Images -> Color patches
    imgs = [Image.fromarray(rgb).resize((patch_size, patch_size)) for rgb in rgbs]
    # pad Patches, make 3x3 grid
    padded_imgs = [
        surround_with_padding(img, round(patch_padding / 2), patch_pad_color)
        for img in imgs
    ]
    grid = image_grid(padded_imgs, 3, 3)

    # pad grid
    if grid_padding > 0:
        grid = surround_with_padding(grid, grid_padding, color=grid_pad_color)

    return grid


def plot_target_with_speaker_order(round_id, data_df):
    round_data = data_df.set_index("round_id").loc[round_id]
    objs = round_data.objs
    speaker_order = round_data.speaker_order
    target = round_data.target
    img = build_grid_image_with_target_highlight(
        objs=objs, order=speaker_order, target_idx=target
    )

    return img


def parse_json_string(json_string):
    """
    Parse a JSON string, handling cases where it may be wrapped in code block formatting.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        dict or None: The parsed JSON object, or None if parsing fails.
    """
    if not isinstance(json_string, str):
        return None
    if json_string.startswith("```json"):
        json_string = re.sub(r"```(json)?", "", json_string)
    try:
        return json.loads(json_string)
    except Exception:
        return None


def parse_annotation_dict(annotation_dict):
    if not annotation_dict or annotation_dict.get("no_reference") == True:
        no_reference = True
        n_spans, spans, span_texts, cells, unique_cells, comments = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
    else:
        spans, span_texts, cells, unique_cells, comments = (
            list(),
            list(),
            list(),
            set(),
            list(),
        )
        no_reference = False
        for ann in annotation_dict["annotations"]:
            spans.append((ann.get("start"), ann.get("end")))
            span_texts.append(ann.get("span"))
            cells.append(ann.get("cells"))
            comments.append(ann.get("comment"))

        n_spans = len(spans)
        unique_cells = set(chain(*cells))

    return n_spans, spans, span_texts, cells, unique_cells, comments, no_reference


def get_unique_cells(model_response):
    if not isinstance(model_response, dict):
        return None
    annotations = model_response.get("annotations")
    if not annotations:
        return None
    unique_cells = set(chain(*[a["cells"] for a in annotations]))
    unique_target_cells = {c for c in unique_cells if c.startswith("t")}
    return unique_target_cells


def unique_cells_iou(ann_series_1, ann_series_2):
    ious = []
    for idx in ann_series_1.index:
        cells_1 = ann_series_1[idx]
        cells_2 = ann_series_2[idx]

        def is_valid_entry(x):
            return isinstance(x, set) and len(x) > 0

        if not (is_valid_entry(cells_1) and is_valid_entry(cells_2)):
            continue

        else:
            intersection = len(cells_1.intersection(cells_2))
            union = len(cells_1.union(cells_2))
            iou = intersection / union
            ious.append(iou)
    return np.mean(ious) if len(ious) > 0 else np.nan



def cells_to_targets(cells_list):
    targets = []
    for c in cells_list or []:
        if isinstance(c, str) and c.startswith("t") and c[1:].isdigit():
            targets.append(int(c[1:]))
    return targets


def parse_record(row, drop_noise_spans=True):
    """
    Parse a row from the model response data, extracting relevant information and filtering based on noise spans.

    Args:
        row (dict): A dictionary representing a row from the model response data.
        drop_noise_spans (bool, optional): Whether to drop annotations with no target cells. Defaults to True.

    Returns:
        dict or None: A dictionary containing parsed information, or None if parsing fails or no valid data is found.
    """

    target_id = row["target_id"]
    game_id, round_num = target_id.rsplit("_", 1)

    parsed = parse_json_string(row["model_response"])
    if not parsed or parsed.get("no_reference") == True:
        return None

    all_targets = []
    for ann in parsed.get("annotations", []):
        cell_list = ann.get("cells") or []
        if drop_noise_spans and len(cell_list) == 0:
            continue
        all_targets.extend(cells_to_targets(cell_list))

    targets = sorted(set(all_targets))

    return {
        "target_id": target_id,
        "game_id": game_id,
        "round_num": round_num,
        "round_id": target_id,
        "targets": targets,
    }


def load_records(filepath, drop_noise_spans=True, restrict_to_condition=None, condition_ids_dict=None):
    """
    Load a model response file, parse responses and optionally restrict to far/split/close conditions.

    Args:
        filepath (str): The path to the model response file.
        drop_noise_spans (bool, optional): Whether to drop annotations with no target cells. Defaults to True.
        restrict_to_condition (str, optional): Condition to restrict the records to 'far', 'split' or 'close' conditions. Defaults to None (i.e., full data).

    Returns:
        tuple: A tuple containing the following elements:
            - records (list): A list of parsed records.
            - model_id (str): Identifier string for the annotation model.
            - evaluated_system (str): The evaluated system identifier.
            - evaluated_system_thinking (bool): Whether the evaluated system is in thinking mode.
            - evaluated_model_label (str): Label (model_id + thinking indicator) for the evaluated model.
            - args (dict): The arguments used for annotation generation.
    """

    with open(filepath, "r") as f:
        file_content = json.load(f)
        model_id = file_content.get("model_id")
        evaluated_system = file_content.get("evaluated_system")
        evaluated_system_thinking = file_content.get("evaluated_enable_thinking")
        evaluated_model_label = osp.split(evaluated_system)[-1] + (
            "-thinking" if evaluated_system_thinking else ""
        )
        args = file_content.get("args")
        responses = file_content.get("responses")

    # raw = pd.read_json(filepath).to_dict(orient='records')
    records = [parse_record(row, drop_noise_spans) for row in responses]

    n_dropped = sum(r is None for r in records)
    records = [r for r in records if r is not None]
    print(
        f"  {len(records)} rounds kept, {n_dropped} dropped (parse failure / no_reference)"
    )
    if restrict_to_condition is not None:
        assert condition_ids_dict is not None, "condition_ids_dict must be provided when restricting to a condition"
        print(f"restricting to condition {restrict_to_condition}")
        condition_ids = condition_ids_dict.get(restrict_to_condition)
        assert condition_ids is not None
        records = [r for r in records if r["round_id"] in condition_ids]
    return records, model_id, evaluated_system, evaluated_system_thinking, evaluated_model_label, args