"""Grounded hidden-state corrections with need, progress and action-fit gates."""

from collections import deque
import os
from pathlib import Path
import uuid

import numpy as np
import torch
from safetensors.torch import load_file

from .. import DATA_ROOT
from ..model.adapter import HiddenAdapter, condition_arrays
from ..observations import CAMERA_KEYS, _project_behavior_b1k_state
from a2meas.resources import verify_file
from .actions import action_summary, guarded_actions
from .client import ReviewError
from .context import request_context
from .gate import ProgressGate
from .grounding import Grounder
from .kinematics import R1ProKinematics
from .review import NeedSelector, FitSelector
from .review_worker import (
    ReviewWorker,
    clear_nonlocal,
    strong_nonlocal,
    screening_negative,
)




class CorrectionController:
    def __init__(self, service, resources, config, task):
        self.service = service
        self.horizon = config["execution_horizon"]
        checkpoint = resources / "adapter/adapter.safetensors"
        verify_file(checkpoint, config["weights"])
        self.adapter = HiddenAdapter().to(service.device)
        self.adapter.load_state_dict(load_file(str(checkpoint)), strict=True)
        self.adapter.eval().requires_grad_(False)
        grounding = resources / "grounding/sam.pt"
        verify_file(grounding, config["grounding"])
        self.grounder = Grounder(grounding, str(service.device))
        data = Path(os.environ.get("OMNIGIBSON_DATA_PATH", DATA_ROOT))
        self.kinematics = R1ProKinematics(data / "omnigibson-robot-assets/models/r1pro")
        self.fit_selector = FitSelector()
        self._review_worker = None
        self.configure_task(task)
        self.reset()

    def configure_task(self, task):
        self.task = task["task"]
        self.definition = task["skill"]
        self.selector = NeedSelector(definitions=[self.definition])
        if self._review_worker is not None:
            self._review_worker.selector = self.selector.fork_for_background()

    def reset(self):
        if self._review_worker is not None:
            self._review_worker.invalidate()
        self.episode_id = uuid.uuid4().hex
        self.progress_gate = ProgressGate(max_unproductive_steps=self.horizon)
        self.history = deque(maxlen=16)
        self.long_frames = deque(maxlen=22)
        self.review_history = deque(maxlen=6)
        self.pending_action = self.previous_state = None
        self.previous_review = self.previous_review_image = (
            self.previous_review_step
        ) = None
        self.blocked_reference = None
        self.fit_approval = self.fit_signature = None
        self.fit_at_step = -1
        self.step = -1
        self._review_prompt = self._last_async_submit_step = None
        self._nonlocal_review_streak = 0

    def close(self):
        if self._review_worker is not None:
            self._review_worker.close()

    def observe(self, observation):
        self.step += 1
        self.state = _project_behavior_b1k_state(observation["robot_r1::proprio"])
        if self.step == 0:
            self.kinematics.verify_observation(observation["robot_r1::proprio"])
        if self.pending_action is not None:
            self.history.append(
                np.concatenate([self.previous_state, self.pending_action])
            )
        self.pending_action = None
        self.previous_state = self.state.copy()
        self.images = np.stack([observation[key][..., :3] for key in CAMERA_KEYS])
        if not self.long_frames or self.step - self.long_frames[-1][0] >= 16:
            self.long_frames.append((self.step, self.images.copy()))

    def action_sent(self, action):
        self.pending_action = action.copy()

    def background_review_inputs(self, kwargs):
        if len(kwargs["frame_steps"]) <= 2:
            return kwargs
        return {
            **kwargs,
            "images": kwargs["images"][[0, -1]],
            "frame_steps": [kwargs["frame_steps"][0], kwargs["frame_steps"][-1]],
        }

    def predict(self, sample):
        flow = self.service.prepare_flow(sample)
        base = self.service.predict_flow(flow)
        if not np.isfinite(base).all():
            raise ValueError("Model returned non-finite actions")
        history = np.zeros((16, 46), np.float32)
        if self.history:
            history[: len(self.history)] = np.stack(self.history)
        history_mask = np.arange(16) < len(self.history)
        kwargs = self._review_inputs(base, history, history_mask, self.task)
        if self._background_review(kwargs, self.task):
            return base, None
        try:
            review = self.selector.infer(**kwargs)
        except ReviewError:
            return self.unavailable(base, "visual_review_unavailable")
        active, reason = self._accept_review(review, self.step, self.images)
        if not active:
            return base, None
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.service.device.type, dtype=self.service.amp_dtype
            ),
        ):
            grounding = self.grounder(self.images, review["target"], self.step)
        try:
            context = request_context(
                self.selector,
                task=self.task,
                skill=self.definition["name"],
                images=self.images,
                step=self.step,
                grounding=grounding,
                history=history,
                history_mask=history_mask,
                base=base,
                review=review,
            )
            if context["confidence"] < 0.7:
                raise ReviewError("Insufficient correction-context confidence")
        except ReviewError:
            return self.unavailable(base, "correction_context_unavailable")
        arrays = condition_arrays(self.task, context, history, history_mask, base)
        batch = {
            key: torch.from_numpy(value).unsqueeze(0).to(self.service.device)
            for key, value in arrays.items()
        }
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.service.device.type, dtype=self.service.amp_dtype
            ),
        ):
            code = self.adapter(batch)
        generated = self.service.predict_flow(
            flow, hidden_residual=lambda hidden: self.adapter.inject(hidden, code)
        )
        delta = generated - base
        if not np.isfinite(delta).all():
            raise ValueError("Adapter returned non-finite actions")
        mask = np.zeros(23, np.float32)
        mask[7:] = 1
        caps = np.full(23, 0.15, np.float32)
        caps[[14, 22]] = 0.25
        proposed = base + np.clip(delta, -caps, caps) * mask
        proposed[:, [14, 22]] = np.clip(proposed[:, [14, 22]], -1, 1)
        summary = action_summary(base, proposed, self.kinematics)
        signature = (
            review["skill_id"],
            review["holding_left"],
            review["holding_right"],
        )
        try:
            if (
                reason.startswith("enter_")
                or self.fit_approval is None
                or signature != self.fit_signature
                or self.step - self.fit_at_step >= 32
            ):
                self.fit_approval = self.fit_selector.infer(
                    task=self.task,
                    images=kwargs["images"][-3:],
                    frame_steps=kwargs["frame_steps"][-3:],
                    state=self.state,
                    history=history,
                    history_mask=history_mask,
                    selected_skill=self.definition,
                    current_review=review,
                    proposal_summary=summary,
                )
                self.fit_signature, self.fit_at_step = signature, self.step
        except ReviewError:
            return self.unavailable(base, "action_fit_unavailable")
        fixed, mask, guard = guarded_actions(
            base, proposed, mask, self.fit_approval, summary
        )
        if not guard["accepted"]:
            self.progress_gate.action_rejected(review, self.step, guard["reason"])
            self.blocked_reference = self.images.copy()
            self.fit_approval = None
            return base, None
        return fixed, mask

    def unavailable(self, base, reason):
        self.progress_gate.release(reason)
        self.progress_gate.reset_pending()
        self.previous_review = None
        self._nonlocal_review_streak = 0
        self.fit_approval = None
        return base, None

    def _accept_review(self, d, step, image):
        had_block = self.progress_gate.blocked_problem
        active, reason = self.progress_gate.update(d, step)
        stable_nonlocal = (
            strong_nonlocal(d)
            and (not active)
            and (self.progress_gate.pending_skill is None)
            and (self.progress_gate.blocked_problem is None)
            and (self.progress_gate.rearm_votes == 0)
        )
        same_prompt = self._review_prompt == self.task
        self._nonlocal_review_streak = (
            min(2, self._nonlocal_review_streak + 1)
            if stable_nonlocal and same_prompt
            else int(stable_nonlocal)
        )
        if self.progress_gate.blocked_problem and (not had_block):
            self.blocked_reference = image.copy()
        if reason == "skip_rearmed_base":
            self.blocked_reference = None
            self.fit_approval = None
        self._review_prompt = self.task
        self.previous_review = d.copy()
        self.previous_review_image = image.copy()
        self.previous_review_step = step
        self.review_history.append({"step": step, **d})
        return (active, reason)

    def _background_review(self, kwargs, policy_prompt):
        """Return True to execute base immediately; False requires a fresh review."""
        gate = self.progress_gate
        worker = self._review_worker
        protected = (
            gate.active_skill is not None
            or gate.pending_skill is not None
            or gate.rearm_votes > 0
            or (self.previous_review is None)
            or (self._review_prompt != policy_prompt)
            or (not gate.blocked_problem and (not clear_nonlocal(self.previous_review)))
        )
        if protected:
            self._nonlocal_review_streak = 0
            if worker is not None:
                worker.invalidate()
            return False
        if worker is None:
            worker = self._review_worker = ReviewWorker(
                self.selector.fork_for_background()
            )
        low_frequency = self._nonlocal_review_streak >= 2 and (not gate.blocked_problem)
        age_limit = 96 if low_frequency else 90
        join_pending = (
            self.step - self.previous_review_step > age_limit
            and worker.future is not None
            and (not worker.future.done())
        )
        delivered = worker.poll(wait=join_pending)
        if delivered is not None:
            age = self.step - delivered["step"]
            invalid = (
                delivered["episode_id"] != self.episode_id
                or delivered["prompt"] != policy_prompt
                or age < 0
                or (age > 90)
                or (delivered["step"] <= gate.last_step)
            )
            if invalid or delivered.get("error"):
                self._nonlocal_review_streak = 0
                worker.invalidate()
                return False
            if not screening_negative(gate, delivered["decision"]):
                self._nonlocal_review_streak = 0
                worker.invalidate()
                return False
            active, _ = self._accept_review(
                delivered["decision"],
                delivered["step"],
                delivered["kwargs"]["images"][-1],
            )
            assert not active
            kwargs = self._review_inputs(
                kwargs["candidate"],
                kwargs["history"],
                kwargs["history_mask"],
                policy_prompt,
            )
        low_frequency = self._nonlocal_review_streak >= 2 and (not gate.blocked_problem)
        age_limit = 96 if low_frequency else 90
        if self.step - self.previous_review_step > age_limit:
            self._nonlocal_review_streak = 0
            worker.invalidate()
            return False
        interval = 96 if low_frequency else 30
        due = (
            self._last_async_submit_step is None
            or self.step - self._last_async_submit_step >= interval
        )
        if due and worker.submit(
            episode_id=self.episode_id,
            step=self.step,
            prompt=policy_prompt,
            kwargs=self.background_review_inputs(kwargs),
        ):
            self._last_async_submit_step = self.step
        return True

    def review_frames(self):
        available = {t: im for t, im in self.long_frames if t <= self.step}
        available[self.step] = self.images
        if self.previous_review_image is not None:
            available[self.previous_review_step] = self.previous_review_image
        selected = set()
        for target in (
            max(0, self.step - 320),
            max(0, self.step - 128),
            self.previous_review_step
            if self.previous_review_step is not None
            else self.step - 32,
            self.step,
        ):
            before = [t for t in available if t <= target]
            selected.add(max(before) if before else min(available))
        steps = sorted(selected)
        return ([available[t] for t in steps], steps)

    def _review_inputs(self, base, history, hm, policy_prompt):
        frames, steps = self.review_frames()
        block = self.progress_gate.blocked_problem
        if block and self.blocked_reference is not None:
            later = [(t, im) for t, im in zip(steps, frames) if t > block["step"]][-3:]
            frames = [self.blocked_reference, *[im for t, im in later]]
            steps = [block["step"], *[t for t, im in later]]
        prior = self.progress_gate.active_skill
        context = {
            "active_skill": prior,
            "previous_review": {
                k: v for k, v in (self.previous_review or {}).items()
                if k not in ("evidence", "holding_left", "holding_right")
            },
            "previous_review_step": self.previous_review_step,
            "recent_reviews": [
                {
                    "step": x["step"],
                    "skill_id": x["skill_id"],
                    "progress": x["progress"],
                }
                for x in list(self.review_history)[-2:]
            ],
            "blocked_problem": block,
            "reference_image_note": "When blocked, the FIRST image triplet is the exhausted interaction; the remaining triplets are later observations.",
            "frame_order": "chronological: longer history, intermediate, previous boundary, current",
            "note": "Past judgments may be wrong; use images to reassess. Steps only label observation order.",
        }
        return dict(
            task=self.task,
            policy_prompt=policy_prompt,
            images=np.stack(frames),
            frame_steps=steps,
            state=self.state,
            history=history,
            history_mask=hm,
            candidate=base,
            gate_context=context,
        )
