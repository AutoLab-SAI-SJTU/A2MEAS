"""Online correction text from grounded regions and observed history."""

import json
import re
import numpy as np
from .client import ReviewError
from .grounding import CAMERAS, image_url


def request_context(
    client, *, task, skill, images, step, grounding, history, history_mask, base, review
):
    fields = {
        "task": task,
        "phase": skill,
        "current_step": int(step),
        "grounding": grounding,
        "executed_prefix": np.asarray(history)[np.asarray(history_mask, bool)][-4:]
        .round(3)
        .tolist(),
        "fresh_unexecuted_proposal": np.asarray(base)[[0, 7, 15, 31]].round(3).tolist(),
    }
    fields["current_visual_diagnosis"] = review
    system = """Form a concise natural-language correction for the current local robot manipulation.
Use the task/phase, current onboard images, verified SAM object regions, actual past
state-action history, and the fresh unexecuted proposal. The task is intent, not
proof of success. Use no expert information or numeric motor commands, angles, or offsets.
Use ONE complete imperative sentence of at most 120 characters. Do not append
explanations or descriptions; finish the sentence before the character limit.
Keep coordinate frames explicit. Do not equate robot-base axes with image axes.
Robot translation directions use R1Pro base_link; never guess camera directions. The smooth gripper command is
-1 fully closed and +1 fully open; These commands do not prove contact or successful grasp.
Unknown spatial relations remain unknown; do not invent contact, depth, grasp or
success. Give a brief imperative instruction plus a qualitative direction with
its reference frame, or null direction if uncertain. Confidence reflects evidence
for this correction. Output JSON only. No numerical control commands."""
    content = [{"type": "text", "text": json.dumps(fields, separators=(",", ":"))}]
    for camera, rgb in zip(CAMERAS, images):
        content.extend(
            [
                {"type": "text", "text": f"camera={camera}, frame={step}"},
                {"type": "image_url", "image_url": {"url": image_url(rgb)}},
            ]
        )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["instruction", "direction", "confidence"],
        "properties": {
            "instruction": {"type": "string", "minLength": 1, "maxLength": 180},
            "direction": {
                "anyOf": [{"type": "string", "maxLength": 100}, {"type": "null"}]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    body = dict(
        model=client.model,
        temperature=0,
        max_tokens=400,
        chat_template_kwargs={"enable_thinking": False},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "correction_context",
                "strict": True,
                "schema": schema,
            },
        },
    )
    result = client.request(body)
    if set(result) != {"instruction", "direction", "confidence"}:
        raise ReviewError("Wrong annotation fields")
    if (
        not isinstance(result["instruction"], str)
        or not 0 < len(result["instruction"]) <= 180
    ):
        raise ReviewError("Instruction length")
    if result["direction"] is not None and (
        not isinstance(result["direction"], str) or len(result["direction"]) > 100
    ):
        raise ReviewError("Direction")
    if (
        type(result["confidence"]) not in (float, int)
        or not 0 <= result["confidence"] <= 1
    ):
        raise ReviewError("Confidence")
    if re.search(r"\d", result["instruction"] + " " + (result["direction"] or "")):
        raise ReviewError("Numeric commands are excluded from recovery text")
    return {**result, "grounding": grounding}
