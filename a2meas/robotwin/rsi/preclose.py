"""One RGB refresh before predicted closing per hand; only shorten native rows."""
import numpy as np


def prefix_length(state, forecast, used):
    """Return original prefix length and newly used hands, without any mutation."""
    rows = np.asarray(forecast)
    if rows.shape != (40, 14) or not np.isfinite(rows).all():
        raise ValueError('Expected original native40x14 rows')
    proposals = []
    for grip in [6, 13]:
        if used[grip] or state[grip] <= .8:
            continue
        closing = np.flatnonzero(rows[:, grip] < .8)
        if len(closing) and int(closing[0]) >= 12:
            proposals.append((int(closing[0]) - 4, grip))
    if not proposals:
        return 40, []
    length = min(n for n, grip in proposals)
    return length, [grip for n, grip in proposals if n == length]


def placement_prefix(actor, native, ctx, *, state=None):
    if not actor.settings.get('observe_before_close', False):
        return native
    sid = ctx.session_id
    memory = actor._sessions[sid]
    used = memory.setdefault('preclose_used', {6: False, 13: False})
    rows = np.asarray(native['actions'], np.float32)
    state = memory['states'][-1] if state is None else state
    length, hands = prefix_length(state, rows, used)
    if not hands:
        return native
    for grip in hands:
        used[grip] = True
    actor._chunk_buffers.pop(sid, None)
    actor._session_chunk_sizes[sid] = length
    actor._try_serve_from_buffer(ctx)
    return {**native, 'actions': rows[:length], 'joints': rows[:length]}


def mug_prefix(actor, native, result, ctx):
    rows = np.asarray(native['actions'], np.float32)
    if rows.shape != (40, 14) or not np.array_equal(rows, np.asarray(result['actions'], np.float32)[:40]):
        return native
    return placement_prefix(actor, native, ctx, state=actor._sessions[ctx.session_id]['state'])
