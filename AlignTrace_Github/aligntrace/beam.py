"""Length-only beam fallback and distinct completion. No biological constraints.
Quicksort routines retain Numba BSD-2-Clause attribution (licenses/).
"""
import numpy as np
from numba import njit

@njit(cache=True,inline='always')
def partition_into(A,R,low,high):
    mid=(low+high)>>1
    if A[R[mid]]<A[R[low]]:R[low],R[mid]=R[mid],R[low]
    if A[R[high]]<A[R[mid]]:R[high],R[mid]=R[mid],R[high]
    if A[R[mid]]<A[R[low]]:R[low],R[mid]=R[mid],R[low]
    pivot=A[R[mid]];R[high],R[mid]=R[mid],R[high]
    i=low;j=high-1
    while True:
        while i<high and A[R[i]]<pivot:i+=1
        while j>=low and pivot<A[R[j]]:j-=1
        if i>=j:break
        R[i],R[j]=R[j],R[i];i+=1;j-=1
    R[i],R[high]=R[high],R[i]
    return i

@njit(cache=True,inline='always')
def insertion_into(A,R,low,high):
    for i in range(low+1,high+1):
        k=R[i];v=A[k];j=i
        while j>low and v<A[R[j-1]]:
            R[j]=R[j-1];j-=1
        R[j]=k
@njit(cache=True)
def _same(a, b, n):
    for i in range(n):
        if a[i] != b[i]:
            return False
    return True

@njit(cache=True)
def _insert(pool, positions, scores, count, sequence, pos, score, target, capacity):
    # Most neighbor proposals cannot enter the current top list. Reject before
    # comparing full strings, which is important at whole-dataset scale.
    if count == capacity and score <= np.min(scores[:count]):
        return count
    duplicate = -1
    for k in range(count):
        if _same(pool[k], sequence, target):
            duplicate = k
            break
    if duplicate >= 0:
        if score > scores[duplicate]:
            scores[duplicate] = score
            positions[duplicate] = pos
        return count
    if count < capacity:
        at = count
        count += 1
    else:
        at = np.argmin(scores[:count])
        if score <= scores[at]:
            return count
    pool[at] = sequence
    positions[at] = pos
    scores[at] = score
    return count

@njit(cache=True)
def _length_heuristic(logp, slots, target):
    """Exact best suffix score for each required emission count, ignoring DNA constraints."""
    h = np.full((slots + 1, target + 1), -1e30, np.float32)
    h[slots, 0] = 0.0
    for s in range(slots - 1, -1, -1):
        emit = max(logp[s, 1], logp[s, 2], logp[s, 3], logp[s, 4])
        h[s, 0] = logp[s, 0] + h[s + 1, 0]
        for need in range(1, min(target, slots - s) + 1):
            h[s, need] = max(logp[s, 0] + h[s + 1, need], emit + h[s + 1, need - 1])
    return h
@njit(cache=True)
def _beam(logp, slots, target, width):
    """Lazy beam selection with a suffix score for the required output length."""
    h = _length_heuristic(logp, slots, target)
    # A new trie node is allocated only for an accepted emitted child.
    # At most width nodes per slot are needed. Zero is the empty prefix.
    capacity = slots * width + 1
    edges = np.empty((capacity, 4), np.int32)
    edges[0] = 0
    seen = np.zeros(capacity, np.int32)
    path_parent = np.empty(capacity, np.int32)
    path_base = np.empty(capacity, np.uint8)
    path_slot = np.empty(capacity, np.int32)
    path_parent[0] = 0
    trie_count = 1
    path_count = 1
    ids = np.zeros(width, np.int32)
    paths = np.zeros(width, np.int32)
    scores = np.full(width, -1e30, np.float32)
    scores[0] = 0
    emitted = np.zeros(width, np.int32)
    alive = 1
    parent = np.empty(width * 5, np.int32)
    actions = np.empty(width * 5, np.uint8)
    counts = np.empty(width * 5, np.int32)
    raw = np.empty(width * 5, np.float32)
    priority = np.empty(width * 5, np.float32)
    order = np.empty(width * 5, np.intp)
    sort_stack = np.empty((2 * width * 5 + 1, 2), np.intp)
    nids = np.zeros_like(ids)
    npaths = np.zeros_like(paths)
    nscores = np.full_like(scores, -1e30)
    nemitted = np.zeros_like(emitted)
    for s in range(slots):
        children = 0
        for j in range(alive):
            for base in range(5):
                count = emitted[j] + (base > 0)
                if count > target or target - count > slots - s - 1:
                    continue
                parent[children] = j
                actions[children] = base
                counts[children] = count
                raw[children] = scores[j] + logp[s, base]
                priority[children] = raw[children] + h[s + 1, target - count]
                children += 1
        if children == 0:
            return (np.zeros((width, target), np.uint8),
                    np.zeros((width, target), np.int32), scores, 0)
        # Traverse the quicksort partition tree right-first and stop at beam capacity.
        for q in range(children):
            order[q] = q
        sort_stack[0, 0] = 0
        sort_stack[0, 1] = children - 1
        pending = 1
        nextalive = 0
        while pending > 0 and nextalive < width:
            pending -= 1
            sortlow = sort_stack[pending, 0]
            sorthigh = sort_stack[pending, 1]
            while sorthigh - sortlow >= 15:
                pivot = partition_into(priority, order, sortlow, sorthigh)
                if pivot > sortlow:
                    sort_stack[pending, 0] = sortlow
                    sort_stack[pending, 1] = pivot - 1
                    pending += 1
                sort_stack[pending, 0] = pivot
                sort_stack[pending, 1] = pivot
                pending += 1
                sortlow = pivot + 1
            insertion_into(priority, order, sortlow, sorthigh)
            for oi in range(sorthigh, sortlow - 1, -1):
                c = order[oi]
                p = parent[c]
                base = actions[c]
                node = ids[p]
                if base > 0:
                    child = edges[node, base - 1]
                    if child == 0:
                        child = trie_count
                        trie_count += 1
                        edges[child] = 0
                        edges[node, base - 1] = child
                    node = child
                if seen[node] == s + 1:
                    continue
                seen[node] = s + 1
                nids[nextalive] = node
                npaths[nextalive] = paths[p]
                if base > 0:
                    path_parent[path_count] = paths[p]
                    path_base[path_count] = base
                    path_slot[path_count] = s
                    npaths[nextalive] = path_count
                    path_count += 1
                nscores[nextalive] = raw[c]
                nemitted[nextalive] = counts[c]
                nextalive += 1
                if nextalive == width:
                    break
        ids, nids = nids, ids
        paths, npaths = npaths, paths
        scores, nscores = nscores, scores
        emitted, nemitted = nemitted, emitted
        alive = nextalive
    seq = np.zeros((width, target), np.uint8)
    pos = np.zeros((width, target), np.int32)
    for j in range(alive):
        path = paths[j]
        for t in range(emitted[j] - 1, -1, -1):
            seq[j, t] = path_base[path]
            pos[j, t] = path_slot[path]
            path = path_parent[path]
    return seq, pos, scores, alive
@njit(cache=True)
def _complete_distinct(logp, slots, target, pool, positions, scores, count, capacity):
    """Reference-free valid DFS completion on one actual emission path.

    This is used only if beam/neighborhood candidates do not fill the list.
    It preserves the blank positions and explores per-base preferences. No
    claim of globally best K is made. At most the first K valid leaves need
    adding, plus duplicates already present in the pool.
    """
    chosen = np.empty(target, np.int32)
    if count:
        chosen[:] = positions[np.argmax(scores[:count])]
    else:
        gains = np.empty(slots, np.float32)
        for s in range(slots):
            gains[s] = max(logp[s, 1], logp[s, 2], logp[s, 3], logp[s, 4]) - logp[s, 0]
        chosen[:] = np.sort(np.argsort(gains)[slots - target:])
    base_score = 0.0
    for s in range(slots):
        base_score += logp[s, 0]
    prefs = np.empty((target, 4), np.int32)
    for t in range(target):
        prefs[t] = np.argsort(logp[chosen[t], 1:5])[::-1] + 1
    seq = np.zeros(target, np.uint8)
    next_choice = np.zeros(target, np.int32)
    depth = 0
    while depth >= 0 and count < capacity:
        if depth == target:
            score = base_score
            for t in range(target):
                score += logp[chosen[t], seq[t]] - logp[chosen[t], 0]
            count = _insert(pool, positions, scores, count, seq, chosen, score, target, capacity)
            depth -= 1
            continue
        if next_choice[depth] == 4:
            next_choice[depth] = 0
            depth -= 1
            continue
        base = prefs[depth, next_choice[depth]]
        next_choice[depth] += 1
        seq[depth] = base
        depth += 1
        if depth < target:
            next_choice[depth] = 0
    return count
