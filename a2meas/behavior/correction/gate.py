"""Local need, progress, release and rearm transitions."""


class ProgressGate:
    def __init__(self, *, max_unproductive_steps=16):
        self.active_skill = None
        self.no_progress_since = None
        self.pending_skill = None
        self.pending_since = None
        self.pending_kind = None
        self.pending_reviews = 0
        self.trial_entry = False
        self.uncertain_reviews = 0
        self.last_step = -1
        self.max_unproductive_steps = max_unproductive_steps
        self.blocked_problem = None
        self.rearm_votes = 0
        self.rearm_key = None
        self.last_local_review = None
        self.unproductive_steps = 0

    def reset_pending(self):
        self.pending_skill = self.pending_since = None
        self.pending_kind = None
        self.pending_reviews = 0

    def release(self, reason):
        self.active_skill = self.no_progress_since = None
        self.uncertain_reviews = 0
        self.trial_entry = False
        return (False, reason)

    def _update_evidence(self, d, step):
        self.last_step = step
        sid, state, progress = (d["skill_id"], d["execution_status"], d["progress"])
        prior = self.active_skill
        local = sid != "none" and d["local_manipulation"] and (d["confidence"] >= 0.7)
        same = d["same_target_as_previous"]
        if not local:
            self.reset_pending()
            return self.release(
                ("exit_" if prior else "skip_") + "unidentified_local_skill"
            )
        resolved = state in ("recovered", "completed") or (
            state == "normal" and progress == "progressing"
        )
        if resolved:
            self.reset_pending()
            return self.release(("exit_" if prior else "skip_") + state)
        possible_need = (
            progress == "uncertain"
            and (d["confidence"] >= 0.7)
            and (d["need_confidence"] >= 0.5)
        )
        kind = (
            "nonprogress"
            if progress == "no_progress"
            else "trial"
            if possible_need
            else None
        )
        if kind:
            if self.pending_skill != sid or not same or self.pending_kind != kind:
                self.pending_skill, self.pending_since = (sid, step)
                self.pending_kind = kind
                self.pending_reviews = 1
            else:
                self.pending_reviews += 1
        else:
            self.reset_pending()
        repeated = (
            self.pending_kind == "nonprogress"
            and self.pending_skill == sid
            and same
            and (self.pending_since is not None)
            and (step - self.pending_since >= 32)
        )
        trial = (
            self.pending_kind == "trial"
            and self.pending_reviews >= 3
            and (self.pending_skill == sid)
            and same
            and (self.pending_since is not None)
            and (step - self.pending_since >= 64)
        )
        visible_need = (
            d["need_correction"]
            and d["need_confidence"] >= 0.65
            and (state in ("failed", "stalled"))
        )
        if prior == sid and same:
            if progress == "uncertain":
                self.uncertain_reviews += 1
                if self.trial_entry or self.uncertain_reviews > 1:
                    self.reset_pending()
                    return self.release("exit_uncertain_progress")
                return (True, "continue_one_review_grace")
            self.trial_entry = False
            self.uncertain_reviews = 0
            if progress == "no_progress":
                if self.no_progress_since is None:
                    self.no_progress_since = step - 16
                if step - self.no_progress_since >= 96:
                    self.reset_pending()
                    return self.release("exit_no_progress")
                return (True, "continue_observed_local_problem")
            self.no_progress_since = None
            if state == "recovering" and d["need_correction"]:
                return (True, "continue_recovery")
            return self.release("exit_progress_resumed")
        self.active_skill = None
        self.no_progress_since = None
        self.uncertain_reviews = 0
        if visible_need or repeated or trial:
            self.active_skill = sid
            self.trial_entry = trial and (not (visible_need or repeated))
            return (
                True,
                "enter_visible_failure"
                if visible_need
                else "enter_repeated_local_nonprogress"
                if repeated
                else "enter_unconfirmed_local_trial",
            )
        return (False, "skip_awaiting_local_need_evidence")

    def block(self, d, step, reason):
        self.blocked_problem = {
            "step": step,
            "target": d["target"],
            "skill_id": d["skill_id"],
            "reason": reason,
        }
        self.rearm_votes = 0
        self.rearm_key = None
        self.reset_pending()
        self.unproductive_steps = 0
        return self.release(reason)

    def update(self, d, step):
        if step <= self.last_step:
            raise ValueError("Review steps must strictly increase")
        previous_step = self.last_step
        if d["local_manipulation"] and d["skill_id"] != "none":
            self.last_local_review = dict(d)
        if self.blocked_problem:
            self.last_step = step
            local = (
                d["local_manipulation"]
                and d["skill_id"] != "none"
                and (d["confidence"] >= 0.7)
            )
            new_target = (
                d["rearm_reason"] == "new_target"
                and local
                and (
                    d["target"].strip().lower()
                    != self.blocked_problem["target"].strip().lower()
                )
            )
            progress = (
                d["rearm_reason"] == "visible_progress"
                and d["progress"] == "progressing"
                and (d["execution_status"] in ("normal", "recovered", "completed"))
            )
            if (new_target or progress) and d["rearm_confidence"] >= 0.8:
                key = (d["rearm_reason"], d["target"].strip().lower())
                self.rearm_votes = self.rearm_votes + 1 if key == self.rearm_key else 1
                self.rearm_key = key
                if self.rearm_votes >= 2:
                    self.blocked_problem = None
                    self.reset_pending()
                    self.unproductive_steps = 0
                    self.release("skip_rearmed_base")
                    return (False, "skip_rearmed_base")
            else:
                self.rearm_votes = 0
                self.rearm_key = None
            return self.release("skip_same_problem_latched")
        if self.active_skill:
            if d["progress"] != "progressing":
                self.unproductive_steps += step - previous_step
                if self.unproductive_steps >= self.max_unproductive_steps:
                    self.last_step = step
                    return self.block(
                        self.last_local_review or d, step, "exit_no_progress"
                    )
            else:
                self.unproductive_steps = 0
        active, reason = self._update_evidence(d, step)
        if reason in ("exit_no_progress", "exit_uncertain_progress"):
            return self.block(d, step, reason)
        if d["progress"] == "progressing" and d["confidence"] >= 0.7:
            self.unproductive_steps = 0
        return (active, reason)

    def action_rejected(self, d, step, reason):
        return self.block(d, step, reason)
