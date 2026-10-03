"""
Active Vision Two-Stage Pipeline: zoom_script.py

Stage 2: Crop zoom regions from the original image, inject them as additional
          visual context, and continue generation for criticism + final answer.

After both stages, the function re-assembles a single DataProto that looks
to the rest of the training loop as if the model produced one long response:

    input_ids  = [prompt1 | stage1_resp | zoom_injection | stage2_resp]
    prompts    = prompt1 tokens (original prompt)
    responses  = stage1_resp + zoom_injection + stage2_resp tokens
    response_mask    = 1s on ALL real response tokens (for reward function)
    multi_modal_data = original images + crop images per sample

This module is called from ray_trainer.py when active_vision.enabled is True.
"""

import re
import json
import os
from PIL import ImageDraw
from copy import deepcopy

import torch
import numpy as np
from tensordict import TensorDict

from ..protocol import DataProto
from ..utils.dataset import process_image


# Cap on how many predicted bboxes from <first_answer> we will crop and
# inject as zoom regions. Beyond this the injection grows unboundedly with
# the model's bbox count, which directly inflates `injection_len` and the
# actor's forward-pass sequence length.
MAX_CROPS_PER_SAMPLE = 4


# Module-level counters accumulated across all training steps. Read by
# ray_trainer at end of training to populate training_summary.txt. The
# trainer driver runs single-process, so a module-level dict is safe;
# Ray rollout/actor workers don't import this module.
_active_vision_stats = {
    "n_calls": 0,                       # how many times zoom_in_on_first_answer was invoked (one per training step)
    "n_samples_processed": 0,           # total trajectories across all steps
    "n_crops_truncated_at_cap": 0,
    "n_samples_overflow_discarded": 0,  # samples that overflowed max_model_len even after cropping → discarded (loss_mask=0)
    "n_samples_center_crop_fallback": 0, # samples that hit the center-crop branch (no valid parsed bbox)
}


def get_active_vision_stats() -> dict:
    """Return a shallow copy of the module-level stats dict. Used by the
    trainer when writing training_summary.txt."""
    return dict(_active_vision_stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_top_level_json_array(content: str):
    """Return the first top-level JSON list parsed from ``content`` by
    walking bracket depth (so nested lists like bbox_2d don't confuse it),
    or [] if none is found / not parseable.
    """
    start_idx = content.find("[")
    if start_idx == -1:
        return []

    depth, in_string, escaped, end_idx = 0, False, False, -1
    for i in range(start_idx, len(content)):
        ch = content[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        return []
    try:
        parsed = json.loads(content[start_idx : end_idx + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def parse_first_answer_bboxes(text):
    """Extract the top-level JSON array from inside <first_answer>...</first_answer>."""
    tag_match = re.search(r"<first_answer>\s*(.*?)\s*</first_answer>", text, re.DOTALL)
    if not tag_match:
        return []
    return _extract_top_level_json_array(tag_match.group(1))


def _iter_all_boxes(obj):
    """Recursively yield [x1, y1, x2, y2] from any bbox nesting format.

    Handles:
      - [{"bbox_2d": [x1,y1,x2,y2]}, ...]
      - [[x1,y1,x2,y2], ...]
      - arbitrarily nested lists of either
    """

    def _is_box(v):
        return isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)

    if isinstance(obj, dict):
        bbox = obj.get("bbox_2d", None)
        if _is_box(bbox):
            yield bbox
        return
    if _is_box(obj):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_all_boxes(item)


def crop_with_annotation(image, bbox, padding_ratio=0.10, max_aspect_ratio=190):
    """Crop a bbox region with slight context padding and draw the bbox boundary on it.

    Returns ``None`` if the resulting crop has an aspect ratio exceeding
    ``max_aspect_ratio`` (Qwen2-VL's ``smart_resize`` rejects ratios >= 200).
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pad_x = int(max(1, x2 - x1) * padding_ratio)
    pad_y = int(max(1, y2 - y1) * padding_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(image.width, x2 + pad_x)
    cy2 = min(image.height, y2 + pad_y)
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    if max(crop_w, crop_h) / max(min(crop_w, crop_h), 1) > max_aspect_ratio:
        return None
    crop = image.crop((cx1, cy1, cx2, cy2))
    if os.environ.get("AV_INJECTION_STYLE", "zoom") != "crop_generic":
        ImageDraw.Draw(crop).rectangle(
            [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline="red", width=3
        )
    return crop


# ---------------------------------------------------------------------------
# Stage 2 prompt builder
# ---------------------------------------------------------------------------

def _extract_query_from_prompt(prompt_text: str) -> str:
    """Extract the query string from the formatted prompt text.

    The prompt is expected to start with:
        Please find "<query>" with bounding boxes ...
    Returns the extracted query, or a generic fallback if parsing fails.
    """
    m = re.search(r'[Pp]lease find\s+"([^"]+)"', prompt_text)
    if m:
        return m.group(1)
    m = re.search(r"[Pp]lease find\s+'([^']+)'", prompt_text)
    if m:
        return m.group(1)
    return "the queried region"


def _get_assistant_open_suffix(processor) -> str:
    """Return the bytes that ``apply_chat_template(..., add_generation_prompt=True)``
    appends to open the next assistant turn for this processor.

    For Qwen3-VL-Instruct this is ``"<|im_start|>assistant\\n"``.
    For Qwen3-VL-Thinking this is ``"<|im_start|>assistant\\n<think>\\n"``.

    Extracted dynamically rather than hard-coded so the inline path matches
    whatever the processor's chat template would produce.
    """
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    idx = rendered.rindex("<|im_start|>assistant")
    return rendered[idx:]


def compute_num_image_pads(processor, processed_crop, min_pixels=None, max_pixels=None):
    """Compute the number of <|image_pad|> tokens an image will need.

    The Qwen3VL processor expands each <|image_pad|> in text into
    ``grid_thw.prod() // merge_size**2`` tokens.  This helper runs the
    image through the image_processor to obtain ``image_grid_thw`` and
    returns that count so we can build injection token IDs that match
    the vision encoder output exactly.

    Args:
        processor: The Qwen3VL processor (has .image_processor).
        processed_crop: A PIL Image that has already been through
                        ``process_image`` (resized, RGB, etc.).

    Returns:
        int: number of <|image_pad|> tokens needed for this image.
    """
    ip = processor.image_processor
    merge_length = ip.merge_size ** 2

    # Run the image processor on a single image to get image_grid_thw
    img_inputs = ip(images=[processed_crop], return_tensors="pt")
    grid_thw = img_inputs["image_grid_thw"]  # (1, 3)
    num_pads = int(grid_thw[0].prod().item() // merge_length)
    return num_pads


def build_zoom_injection_text(
    crops,
    valid_bboxes,
    query: str = "the queried region",
    num_pads_per_crop: list[int] | None = None,
    processor=None,
):
    """Build the chat-template-correct injection text for stage 2.

    The returned text is meant to be concatenated *immediately after* the
    stage-1 generated response. It:

      1. closes the stage-1 assistant turn with ``<|im_end|>``,
      2. opens a fresh user turn whose content is the zoom block + Step 5/6
         instructions (this is where the ``<|vision_start|>...<|vision_end|>``
         markers live — vision tokens belong in user turns),
      3. closes that user turn with ``<|im_end|>``,
      4. opens a new assistant turn with the same suffix that
         ``apply_chat_template(..., add_generation_prompt=True)`` would emit.

    When ``num_pads_per_crop`` is provided each crop placeholder is expanded
    to the correct number of ``<|image_pad|>`` tokens so the token count
    matches the vision-encoder features produced by
    ``_process_multi_modal_inputs``.  When ``None`` a single ``<|image_pad|>``
    is used (suitable for the *processor* which expands them itself, but NOT
    for direct tokenizer encoding).

    Args:
        crops: list of PIL crop images (one per predicted bbox).
        valid_bboxes: list of [x1,y1,x2,y2] for each crop, or None.
        query: the user's original query string (e.g. "head of the dog")
               used in the Step 5/6 instructions so the model knows what
               the boxes should enclose.
        num_pads_per_crop: list of ints giving the number of <|image_pad|>
                          tokens per crop image.  Must be same length as
                          ``crops``.  If *None*, 1 pad per crop is used.
        processor: HF processor — used to derive the assistant-open suffix
                   for the model variant. If None, falls back to the
                   Instruct-style suffix (``<|im_start|>assistant\\n``).
    """
    is_center_crop = valid_bboxes is None
    generic_style = os.environ.get("AV_INJECTION_STYLE", "zoom") == "crop_generic"
    tag = "crop" if generic_style else "zoom"

    if is_center_crop:
        zoom_lines = [
            f"<{tag}>",
            "No valid bounding boxes could be parsed from your <first_answer>. "
            "Here is a center-crop of the original image for additional context:",
        ]
    else:
        zoom_lines = [
            f"<{tag}>",
            "Here are zoomed-in crops of your predicted bounding box regions from the original image:",
        ]
    for j in range(len(crops)):
        if not is_center_crop and j < len(valid_bboxes):
            b = valid_bboxes[j]
            zoom_lines.append(
                f"Region {j + 1} (original image coordinates: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]):"
            )
        else:
            zoom_lines.append(f"Region {j + 1} (center crop — no bbox was parsed):")
        n_pads = num_pads_per_crop[j] if num_pads_per_crop is not None else 1
        pad_str = "<|image_pad|>" * n_pads
        zoom_lines.append(f"<|vision_start|>{pad_str}<|vision_end|>")
    zoom_lines.append(f"</{tag}>")

    # Step 5 and Step 6 instructions — mirrored from the full prompt
    # but grounded with the actual query and a reminder about coordinates.
    # When center-crop fallback was used, the instructions reflect that
    # the model is seeing a center-zoom rather than its own predictions.
    if is_center_crop:
        step5_instruction = (
            f'In <criticism> </criticism> tags, examine the center-crop of the original image above '
            f'and re-evaluate your <first_answer>. Note: no valid bounding boxes could be parsed from '
            f'your first answer, so this is a center-zoom of the image rather than a crop of your '
            f'prediction. Do the boxes tightly enclose "{query}"? If adjustments are needed, describe '
            f'the issue and suggested adjustments. Examples of necessary adjustments could be to make '
            f'the bboxes smaller or bigger, or move the bboxes in any direction.'
        )
    else:
        step5_instruction = (
            f'In <criticism> </criticism> tags, examine the zoomed-in crops above and check your '
            f'<first_answer>. Do the boxes tightly enclose "{query}"? If adjustments are needed, '
            f'describe the issue and suggested adjustments. Examples of necessary adjustments could '
            f'be to make the bboxes smaller or bigger, or move the bboxes in any direction.'
        )

    zoom_lines += [
        "",
        "IMPORTANT: The cropped images above are extra context only.",
        "Always propose bounding boxes using coordinates from the ORIGINAL image,",
        "not relative to the zoomed crops.",
        "",
        "Now continue with Step 5 and Step 6.",
        "",
        "STEP 5 — Criticism (Self-Check)",
        step5_instruction,
        "The last part inside <criticism> MUST be exactly one of:",
        '- "ADJUSTMENT: YES"',
        '- or "ADJUSTMENT: NO"',
        f'Output "ADJUSTMENT: NO" if the boxes and points in <first_answer> are correct and no adjustments are necessary. Output "ADJUSTMENT: YES" if the <first_answer> needs adjustments to enclose "{query}".',
        *(
            [
                "Format (describe what YOU actually observe in the crops; do not copy this wording):",
                '- "<criticism>[your assessment of whether the boxes tightly enclose the target, '
                'and the specific adjustment you will make, if any]. ADJUSTMENT: YES|NO</criticism>"',
            ]
            if generic_style
            else [
                "Examples:",
                '- "<criticism>...The box is already tight and correctly placed. ADJUSTMENT: NO</criticism>"',
                '- "<criticism>...The box is slightly too large and shifted down; I will move it up and shrink it. ADJUSTMENT: YES</criticism>"',
            ]
        ),
        "",
        "STEP 6 — FINAL ANSWER",
        "In <answer> </answer> tags, output the final answer.",
        '- If <criticism> ends with "ADJUSTMENT: NO":',
        "  - The JSON list in <answer> MUST be IDENTICAL to the JSON list in <first_answer>.",
        '- If <criticism> ends with "ADJUSTMENT: YES":',
        "  - At least one bbox_2d or point_2d in <answer> MUST be different from those in <first_answer>,",
        "    and the changes should address the issues described in <criticism>.",
        "The <answer> should contain a JSON list of entries with bbox_2d and point_2d.",
        '[{"bbox_2d": [qx1, qy1, qx2, qy2], "point_2d": [qcx, qcy]},...]',
        "\nOUTPUT FORMAT EXAMPLE (STRUCTURE ONLY):",
        "<criticism>criticism here. ADJUSTMENT: YES|NO</criticism>",
        "<answer>[{\"bbox_2d\": [qx1, qy1, qx2, qy2], \"point_2d\": [qcx, qcy]},...]</answer>",
    ]
    user_turn_content = "\n".join(zoom_lines)

    if processor is not None:
        assistant_open = _get_assistant_open_suffix(processor)
    else:
        assistant_open = "<|im_start|>assistant\n"

    # Close stage-1 assistant turn, open new user turn with the zoom block
    # and Step 5/6 instructions, close it, open the stage-2 assistant turn.
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_turn_content}<|im_end|>\n"
        f"{assistant_open}"
    )


# ---------------------------------------------------------------------------
# Main entry point — called from RayPPOTrainer._make_batch_data
# ---------------------------------------------------------------------------

def zoom_in_on_first_answer(trainer, gen_batch: DataProto) -> DataProto:
    """Two-stage generation with active-vision zoom, returning a re-assembled
    DataProto whose ``responses`` contains the full concatenated output from
    both stages (with zoom injection in between).

    The returned DataProto has the following structure:

        prompts          = original Stage 1 prompt tokens
        responses        = [stage1_resp | zoom_injection | stage2_resp]
        input_ids        = [prompts | responses]
        response_mask    = 1 on ALL real response tokens (stage1 + injection + stage2),
                           0 after EOS — used by reward function
        position_ids     = continuous across full sequence (mRoPE-aware)
        multi_modal_data = original images + crop images per sample

    Args:
        trainer: The RayPPOTrainer instance.
        gen_batch: DataProto with input_ids, attention_mask, position_ids,
                   raw_prompt_ids, multi_modal_data.

    Returns:
        DataProto with the re-assembled training batch.
    """
    config = trainer.config
    av_config = config.active_vision
    crop_padding = av_config.crop_padding_ratio
    if _active_vision_stats.get("n_calls", 0) == 0:
        print(
            f"[active_vision] mask_stage1_loss={av_config.mask_stage1_loss}  "
            f"(Phase 1 if True; stage-1 tokens get zero PG gradient, "
            f"only KL-to-reference regularisation)"
        )

    original_multi_modal_data = gen_batch.non_tensor_batch["multi_modal_data"]
    original_raw_prompt_ids = gen_batch.non_tensor_batch["raw_prompt_ids"]

    # Save original prompt tensors (before Stage 1 modifies anything).
    # These are left-padded: (num_orig_samples, prompt_length)
    orig_prompt_input_ids = gen_batch.batch["input_ids"].clone()
    orig_prompt_attention_mask = gen_batch.batch["attention_mask"].clone()
    orig_prompt_position_ids = gen_batch.batch["position_ids"].clone()

    # ---------------------------------------------------------------
    # Stage 1: until </first_answer>
    # ---------------------------------------------------------------

    print("[zoom] Stage 1 generation starting...")
    stage1_batch = deepcopy(gen_batch)
    stage1_output = trainer.actor_rollout_ref_wg.generate_sequences(stage1_batch)
    del stage1_batch

    print("[zoom] Stage 1 done.")

    # stage1_output.batch has: prompts, responses, input_ids, attention_mask,
    # response_mask, position_ids — all with Stage 1 prompt + Stage 1 response.
    stage1_response_ids = stage1_output.batch["responses"]  # (batch_size, stage1_resp_len)
    stage1_response_mask = stage1_output.batch["response_mask"]  # 1s up to EOS, 0s after

    eos_token_id = stage1_output.meta_info["eos_token_id"]

    stage1_texts = [
        trainer.tokenizer.decode(ids, skip_special_tokens=False)
        for ids in stage1_response_ids
    ]
    del stage1_output

    batch_bboxes = [parse_first_answer_bboxes(t) for t in stage1_texts]

    n = config.worker.rollout.n  # rollout responses per original sample
    num_orig = len(original_multi_modal_data)
    batch_size = len(stage1_texts)  # = num_orig * n

    # ---------------------------------------------------------------
    # Cache processed original images once per unique sample
    # ---------------------------------------------------------------
    orig_first_img = {}   # orig_idx -> PIL Image (for cropping)
    orig_imgs_proc = {}   # orig_idx -> [process_image(img) ...]
    orig_imgs_raw = {}    # orig_idx -> list(images)
    for orig_idx in range(num_orig):
        imgs = list(original_multi_modal_data[orig_idx]["images"])
        orig_imgs_raw[orig_idx] = imgs
        orig_first_img[orig_idx] = process_image(
            imgs[0], config.data.min_pixels, config.data.max_pixels
        )
        orig_imgs_proc[orig_idx] = [
            process_image(img, config.data.min_pixels, config.data.max_pixels)
            for img in imgs
        ]

    # ---------------------------------------------------------------
    # Crop each trajectory's predicted bbox regions
    # ---------------------------------------------------------------
    cropped_images_per_sample = []
    cropped_bboxes_per_sample = []
    for i, bboxes in enumerate(batch_bboxes):
        orig_img = orig_first_img[i // n]
        crops, valid_bboxes = [], []
        for bbox in _iter_all_boxes(bboxes):
            if len(crops) >= MAX_CROPS_PER_SAMPLE:
                # Cap: extra bboxes are dropped to bound `injection_len`. See
                # MAX_CROPS_PER_SAMPLE comment near top of file.
                _active_vision_stats["n_crops_truncated_at_cap"] += 1
                break
            x1 = max(0, min(int(bbox[0]), orig_img.width))
            y1 = max(0, min(int(bbox[1]), orig_img.height))
            x2 = max(0, min(int(bbox[2]), orig_img.width))
            y2 = max(0, min(int(bbox[3]), orig_img.height))
            if x2 > x1 and y2 > y1:
                crop = crop_with_annotation(orig_img, [x1, y1, x2, y2], padding_ratio=crop_padding)
                if crop is not None:
                    crops.append(crop)
                    valid_bboxes.append([x1, y1, x2, y2])
        if not crops:
            cx, cy = orig_img.width // 2, orig_img.height // 2
            r = min(orig_img.width, orig_img.height) // 4
            crops = [orig_img.crop((cx - r, cy - r, cx + r, cy + r))]
            valid_bboxes = None
        cropped_images_per_sample.append(crops)
        cropped_bboxes_per_sample.append(valid_bboxes)

    # ---------------------------------------------------------------
    # Detect mRoPE
    # ---------------------------------------------------------------
    use_mrope = False
    if trainer.processor is not None and "Qwen2VLImageProcessor" in trainer.processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in trainer.processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index
        use_mrope = True

    # ---------------------------------------------------------------
    # Decode original prompts
    # ---------------------------------------------------------------
    original_prompt_texts = [
        trainer.tokenizer.decode(ids, skip_special_tokens=False)
        for ids in original_raw_prompt_ids
    ]

    # Extract the query string from each original prompt for Step 5/6 instructions
    original_queries = [_extract_query_from_prompt(t) for t in original_prompt_texts]

    # ---------------------------------------------------------------
    # Build Stage 2 prompts and tokenize for vLLM generation
    # ---------------------------------------------------------------
    stage2_input_ids_list = []
    stage2_attn_mask_list = []
    stage2_position_ids_list = []
    stage2_raw_prompt_ids_list = []
    stage2_multi_modal_data = []
    stage2_image_grid_thw_list = []

    zoom_injection_token_counts = []

    # Track the number of <|image_pad|> tokens per crop for each sample,
    # needed during re-assembly to build input_ids with correct image pad counts.
    num_pads_per_sample = []

    print("[zoom] Building Stage 2 prompts...")
    # Overflow guard: vLLM rejects a stage-2 prompt if its length plus the
    # generation budget exceeds max_model_len. Compute the budget once here.
    max_model_len = config.worker.rollout.max_model_len
    stage2_resp_max = config.worker.rollout.response_length
    overflow_budget = max_model_len - stage2_resp_max  # max allowed prompt length

    discarded_per_sample = [False] * batch_size

    def _build_one_stage2_sample(crops_arg, valid_bboxes_arg, response_text_arg, orig_idx_arg, query_arg):
        """Compute all per-sample artefacts for one stage-2 sample."""
        crop_imgs_proc = [
            process_image(c, config.data.min_pixels, config.data.max_pixels)
            for c in crops_arg
        ]
        crop_num_pads = [
            compute_num_image_pads(trainer.processor, crop_proc)
            for crop_proc in crop_imgs_proc
        ]
        continuation_text = build_zoom_injection_text(
            crops_arg, valid_bboxes_arg, query=query_arg, processor=trainer.processor,
        )
        full_stage2_text = original_prompt_texts[orig_idx_arg] + response_text_arg + continuation_text
        zoom_injection_ids = trainer.tokenizer.encode(continuation_text, add_special_tokens=False)
        stage2_raw_prompt_ids = trainer.tokenizer.encode(full_stage2_text, add_special_tokens=False)
        all_processed_images = orig_imgs_proc[orig_idx_arg] + crop_imgs_proc

        model_inputs = trainer.processor(
            all_processed_images, [full_stage2_text],
            add_special_tokens=False, return_tensors="pt",
        )
        input_ids_i = model_inputs.pop("input_ids")[0]
        attention_mask_i = model_inputs.pop("attention_mask")[0]

        if use_mrope:
            vision_position_ids = get_rope_index(
                trainer.processor,
                input_ids=input_ids_i,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask_i,
            )
            position_ids_i = torch.cat(
                (torch.arange(len(input_ids_i), device=input_ids_i.device).unsqueeze(0), vision_position_ids),
                dim=0,
            )
        else:
            position_ids_i = torch.clip(attention_mask_i.cumsum(dim=0) - 1, min=0)

        return dict(
            crop_num_pads=crop_num_pads,
            zoom_injection_ids=zoom_injection_ids,
            stage2_raw_prompt_ids=stage2_raw_prompt_ids,
            input_ids_i=input_ids_i,
            attention_mask_i=attention_mask_i,
            position_ids_i=position_ids_i,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
        )

    for i, response_text in enumerate(stage1_texts):
        orig_idx = i // n
        crops = cropped_images_per_sample[i]
        valid_bboxes = cropped_bboxes_per_sample[i]
        query = original_queries[orig_idx]
        if valid_bboxes is None:
            _active_vision_stats["n_samples_center_crop_fallback"] += 1

        s2 = _build_one_stage2_sample(crops, valid_bboxes, response_text, orig_idx, query)

        if s2["input_ids_i"].size(0) > overflow_budget:
            old_len = s2["input_ids_i"].size(0)
            s2 = _build_one_stage2_sample([], None, response_text, orig_idx, query)
            new_len = s2["input_ids_i"].size(0)

            cropped_images_per_sample[i] = []
            cropped_bboxes_per_sample[i] = None
            crops, valid_bboxes = [], None

            discarded_per_sample[i] = True
            _active_vision_stats["n_samples_overflow_discarded"] += 1

            if new_len > overflow_budget:
                # Even a no-crop injection overflows — the prompt or stage-1
                # response is the cause. vLLM will reject this batch.
                print(
                    f"[zoom] FATAL OVERFLOW sample {i}: len={new_len} > budget={overflow_budget} "
                    f"even with no crops. Bump worker.rollout.max_model_len.",
                    flush=True,
                )
            else:
                print(
                    f"[zoom] OVERFLOW sample {i}: {old_len} → {new_len} (budget {overflow_budget}). "
                    f"Rebuilt with no crops, marked discarded (no gradient, no reward).",
                    flush=True,
                )

        num_pads_per_sample.append(s2["crop_num_pads"])
        zoom_injection_token_counts.append(len(s2["zoom_injection_ids"]))
        stage2_raw_prompt_ids_list.append(s2["stage2_raw_prompt_ids"])
        stage2_input_ids_list.append(s2["input_ids_i"])
        stage2_attn_mask_list.append(s2["attention_mask_i"])
        stage2_position_ids_list.append(s2["position_ids_i"])
        stage2_multi_modal_data.append({"images": orig_imgs_raw[orig_idx] + crops})
        stage2_image_grid_thw_list.append(s2["image_grid_thw"])

    # ---------------------------------------------------------------
    # Left-pad and stack Stage 2 prompts into batch tensors for vLLM
    # ---------------------------------------------------------------
    max_len = max(ids.size(-1) for ids in stage2_input_ids_list)

    padded_input_ids = torch.full((batch_size, max_len), trainer.tokenizer.pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_position_ids = torch.zeros(
        (batch_size, 4, max_len) if use_mrope else (batch_size, max_len), dtype=torch.long
    )
    for i in range(batch_size):
        seq_len = stage2_input_ids_list[i].size(-1)
        padded_input_ids[i, max_len - seq_len:] = stage2_input_ids_list[i]
        padded_attention_mask[i, max_len - seq_len:] = stage2_attn_mask_list[i]
        padded_position_ids[i, ..., max_len - seq_len:] = stage2_position_ids_list[i]

    stage2_batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "position_ids": padded_position_ids,
            },
            batch_size=batch_size,
        ),
        non_tensor_batch={
            "raw_prompt_ids": np.array(stage2_raw_prompt_ids_list, dtype=object),
            "multi_modal_data": np.array(stage2_multi_modal_data, dtype=object),
        },
        meta_info=dict(gen_batch.meta_info),
    )
    # n trajectories already exist from Stage 1; Stage 2 generates 1 continuation each
    stage2_batch.meta_info["n"] = 1

    # ---------------------------------------------------------------
    # Stage 2: generate criticism + final answer
    # ---------------------------------------------------------------
    print("[zoom] Stage 2 generation starting...")
    stage2_output = trainer.actor_rollout_ref_wg.generate_sequences(stage2_batch)
    print("[zoom] Stage 2 done.")

    # stage2_output.batch has: prompts (= stage2 full prompt), responses (= stage2 resp),
    # input_ids (= stage2 prompt + stage2 resp), attention_mask, response_mask, position_ids
    stage2_response_ids = stage2_output.batch["responses"]       # (batch_size, stage2_resp_len)
    stage2_response_mask = stage2_output.batch["response_mask"]  # standard: 1s up to EOS

    zoom_injection_ids_per_sample = []
    for i in range(batch_size):
        orig_idx = i // n
        crops = cropped_images_per_sample[i]
        valid_bboxes = cropped_bboxes_per_sample[i]
        query = original_queries[orig_idx]
        injection_text = build_zoom_injection_text(
            crops, valid_bboxes, query=query,
            num_pads_per_crop=num_pads_per_sample[i],
            processor=trainer.processor,
        )
        injection_ids = trainer.tokenizer.encode(injection_text, add_special_tokens=False)
        zoom_injection_ids_per_sample.append(torch.tensor(injection_ids, dtype=torch.long))

    # Compute the maximum combined response length
    stage1_resp_len = stage1_response_ids.size(1)
    stage2_resp_len = stage2_response_ids.size(1)
    max_injection_len = max(t.size(0) for t in zoom_injection_ids_per_sample)
    combined_resp_len = stage1_resp_len + max_injection_len + stage2_resp_len

    # Build combined responses tensor: [stage1_resp | injection | stage2_resp]
    # Right-padded with pad_token_id
    pad_id = trainer.tokenizer.pad_token_id
    combined_responses = torch.full((batch_size, combined_resp_len), pad_id, dtype=torch.long)
    combined_response_mask = torch.zeros((batch_size, combined_resp_len), dtype=torch.long)
    combined_loss_mask = torch.zeros((batch_size, combined_resp_len), dtype=torch.long)

    for i in range(batch_size):
        if discarded_per_sample[i]:
            # Sample overflowed max_model_len and was rebuilt with no crops
            # for stage-2 vLLM. Don't write any real content into combined_*
            # tensors — they remain at their init values (pad_id / 0 / 0).
            # Effect downstream:
            #   - response_mask = 0 -> reward worker sees response_length=0.
            #   - actor_loss_mask = 0 -> no gradient flows from this slot.
            # The sample slot still exists in the batch (preserves shape and
            # GRPO group cardinality).
            continue

        injection_ids = zoom_injection_ids_per_sample[i]
        injection_len = injection_ids.size(0)

        # Valid lengths of LLM-generated portions (tokens before EOS padding)
        s1_valid = int(stage1_response_mask[i].sum().item())
        s2_valid = int(stage2_response_mask[i].sum().item())

        real_ids = torch.cat([
            stage1_response_ids[i, :s1_valid],
            injection_ids,
            stage2_response_ids[i, :s2_valid],
        ])
        real_len = real_ids.size(0)  # = s1_valid + injection_len + s2_valid

        combined_responses[i, :real_len] = real_ids
        # Tail [real_len : combined_resp_len] stays at pad_id from the
        # torch.full init above.

        # response_mask: 1 on the entire contiguous real run, 0 on trailing pad.
        combined_response_mask[i, :real_len] = 1

        # actor_loss_mask: 1 on LLM-generated tokens (stage1_real, stage2_real),
        # 0 on injection (we didn't generate it) and trailing pad.
        if not av_config.mask_stage1_loss:
            combined_loss_mask[i, :s1_valid] = 1
        combined_loss_mask[i, s1_valid + injection_len : s1_valid + injection_len + s2_valid] = 1

    # ---------------------------------------------------------------
    # Build the prompt part — we need the original Stage 1 prompt tokens.
    # After Stage 1 generation with n>1, the prompts were repeated.
    # We use orig_prompt_input_ids repeated by n.
    # ---------------------------------------------------------------
    if n > 1:
        prompt_ids = orig_prompt_input_ids.repeat_interleave(n, dim=0)
        prompt_attn = orig_prompt_attention_mask.repeat_interleave(n, dim=0)
        prompt_pos = orig_prompt_position_ids.repeat_interleave(n, dim=0)
    else:
        prompt_ids = orig_prompt_input_ids
        prompt_attn = orig_prompt_attention_mask
        prompt_pos = orig_prompt_position_ids

    prompt_len = prompt_ids.size(1)

    # Full sequence: [prompt | combined_response]
    full_input_ids = torch.cat([prompt_ids, combined_responses], dim=1)
    full_seq_len = full_input_ids.size(1)

    # Attention mask: [prompt_attn | combined_response_mask]
    # (response_mask serves as attention_mask for the response portion — 1 on real tokens)
    full_attention_mask = torch.cat([prompt_attn, combined_response_mask], dim=1)

    # Position IDs.
    # - mRoPE path (Qwen2/3-VL): the response section contains crop image
    #   tokens (`<|image_pad|>`s in the injection block). They need the
    #   same (t=0, h=row, w=col) spatial encoding the rollout-side mRoPE
    #   call used. We recompute positions for the
    #   FULL `full_input_ids` (prompt + compacted response) per sample
    #   via get_rope_index, then prepending the standard 1D text counter
    #   as the 4th axis.
    # - Non-mRoPE path: keep the original sequential-delta logic.
    if use_mrope:
        full_position_ids = torch.zeros(
            (batch_size, 4, full_seq_len), dtype=torch.long, device=full_input_ids.device,
        )
        for i in range(batch_size):
            vision_pos_i = get_rope_index(
                trainer.processor,
                input_ids=full_input_ids[i],
                image_grid_thw=stage2_image_grid_thw_list[i],
                video_grid_thw=None,
                attention_mask=full_attention_mask[i],
            )  # (3, full_seq_len) — (t, h, w)
            # 4th axis: standard 1D text counter, padding-aware via cumsum.
            text_pos_i = torch.clamp(
                full_attention_mask[i].long().cumsum(0) - 1, min=0,
            )  # (full_seq_len,)
            full_position_ids[i] = torch.cat(
                [text_pos_i.unsqueeze(0), vision_pos_i], dim=0,
            )
    else:
        resp_position_delta = torch.arange(1, combined_resp_len + 1, device=prompt_pos.device)
        resp_position_delta = resp_position_delta.view(1, -1).expand(batch_size, -1)
        resp_position_ids = prompt_pos[..., -1:] + resp_position_delta
        full_position_ids = torch.cat([prompt_pos, resp_position_ids], dim=-1)

    # ---------------------------------------------------------------
    # Build multi_modal_data (original + crops per sample)
    # ---------------------------------------------------------------
    final_multi_modal_data = np.array(stage2_multi_modal_data, dtype=object)

    # ---------------------------------------------------------------
    # Assemble final DataProto
    # ---------------------------------------------------------------
    final_batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": combined_responses,
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "response_mask": combined_response_mask,
            "actor_loss_mask": combined_loss_mask,
            "position_ids": full_position_ids,
        },
        batch_size=batch_size,
    )

    final_meta = dict(gen_batch.meta_info)
    final_meta["eos_token_id"] = eos_token_id

    final_proto = DataProto(
        batch=final_batch,
        non_tensor_batch={"multi_modal_data": final_multi_modal_data},
        meta_info=final_meta,
    )

    print(
        f"[zoom] Re-assembled DataProto: batch_size={batch_size}, "
        f"prompt_len={prompt_len}, stage1_resp_len={stage1_resp_len}, "
        f"max_injection_len={max_injection_len}, stage2_resp_len={stage2_resp_len}, "
        f"combined_resp_len={combined_resp_len}, full_seq_len={full_seq_len}"
    )

    # Sanity-check: count image_pad tokens in full_input_ids vs expected from multi_modal_data
    image_token_id = trainer.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for i in range(min(batch_size, 2)):  # log first 2 samples
        n_image_pad = (full_input_ids[i] == image_token_id).sum().item()
        n_images = len(final_multi_modal_data[i]["images"])
        pads_in_prompt = (prompt_ids[i] == image_token_id).sum().item()
        pads_in_resp = (combined_responses[i] == image_token_id).sum().item()
        print(
            f"[zoom] Sample {i}: n_images={n_images}, "
            f"image_pad_in_prompt={pads_in_prompt}, image_pad_in_resp={pads_in_resp}, "
            f"total_image_pad={n_image_pad}"
        )

    # Per-step stats accumulate into the module-level dict, read at end of
    # training by ray_trainer to write training_summary.txt. Print a one-line
    # roll-up here so you can see the rate live in the training log.
    _active_vision_stats["n_calls"] += 1
    _active_vision_stats["n_samples_processed"] += batch_size
    print(
        f"[zoom-stats] step_calls={_active_vision_stats['n_calls']} "
        f"samples_this_step={batch_size} "
        f"discarded_total={_active_vision_stats['n_samples_overflow_discarded']} "
        f"center_crop_fallback_total={_active_vision_stats['n_samples_center_crop_fallback']} "
        f"crops_truncated_at_cap_total={_active_vision_stats['n_crops_truncated_at_cap']}",
        flush=True,
    )

    torch.cuda.empty_cache()
    return final_proto
