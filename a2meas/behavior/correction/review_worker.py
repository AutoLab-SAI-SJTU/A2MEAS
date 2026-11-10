"""One bounded background review; positive decisions require current images."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from .client import ReviewError


class ReviewWorker:
    """One bounded request with reset-safe result delivery."""

    def __init__(self, selector):
        self.selector = selector
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="adapter-review"
        )
        self.future = None
        self.generation = 0
        self.closed = False

    def invalidate(self):
        self.generation += 1
        if self.future is not None and self.future.cancel():
            self.future = None

    def submit(self, *, episode_id, step, prompt, kwargs):
        if self.closed or self.future is not None:
            return False
        snapshot = deepcopy(
            dict(
                episode_id=episode_id,
                step=step,
                prompt=prompt,
                generation=self.generation,
                kwargs=kwargs,
            )
        )
        self.future = self.executor.submit(self._run, snapshot)
        return True

    def _run(self, snapshot):
        try:
            snapshot["decision"] = self.selector.infer(**snapshot["kwargs"])
        except ReviewError as exc:
            snapshot["error"] = type(exc).__name__
        return snapshot

    def poll(self, *, wait=False):
        """Consume the request, joining it at the freshness boundary if needed."""
        if self.future is None or (not wait and not self.future.done()):
            return None
        future, self.future = self.future, None
        if future.cancelled():
            return None
        result = future.result()
        return result if result["generation"] == self.generation else None

    def close(self):
        self.closed = True
        self.invalidate()
        self.executor.shutdown(wait=False, cancel_futures=True)


def clear_nonlocal(review):
    """Only a confident nonlocal negative enables asynchronous base execution."""
    return bool(
        review
        and review["skill_id"] == "none"
        and not review["local_manipulation"]
        and not review["need_correction"]
        and review["confidence"] >= 0.70
        and review["execution_status"] in ("normal", "recovered", "completed")
    )


def strong_nonlocal(review):
    """Only repeated confident nonlocal negatives can slow background polling."""
    return bool(
        clear_nonlocal(review)
        and review["confidence"] >= 0.85
        and review["need_confidence"] >= 0.85
    )


def screening_negative(gate, review):
    """Never use a delayed result to enter correction or cast a rearm vote."""
    if gate.blocked_problem:
        return not (
            review["rearm_reason"] in ("new_target", "visible_progress")
            or review["progress"] == "progressing"
        )
    return clear_nonlocal(review)
