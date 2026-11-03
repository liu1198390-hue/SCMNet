"""Decode gland instances with a learned contact barrier and identity consistency."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 4), (4, 0), (4, 4), (4, -4))


def decode_instances(
    region_probability,
    relation_probability,
    contact_probability,
    embedding,
    candidate_threshold=0.35,
    seed_threshold=0.65,
    edge_threshold=0.5,
    contact_threshold=0.65,
    similarity_threshold=0.45,
    min_area=100,
):
    region = np.asarray(region_probability, dtype=np.float32)
    relation = np.asarray(relation_probability, dtype=np.float32)
    contact = np.asarray(contact_probability, dtype=np.float32)
    z = np.asarray(embedding, dtype=np.float32)
    if region.ndim != 2 or contact.shape != region.shape or relation.shape != (8, *region.shape):
        raise ValueError("Expected HxW region/contact and 8xHxW relation")
    if z.ndim != 3 or z.shape[1:] != region.shape:
        raise ValueError("Expected DxHxW identity embedding")
    if not 0 <= candidate_threshold <= seed_threshold <= 1:
        raise ValueError("Require 0 <= candidate_threshold <= seed_threshold <= 1")
    if not 0 <= contact_threshold <= 1 or not -1 <= similarity_threshold <= 1:
        raise ValueError("Contact or similarity threshold outside its probability range")
    height, width = region.shape
    # The contact channel is explicitly trained on background separating nearby glands.
    candidate = (region >= candidate_threshold) & (contact < contact_threshold)
    pixel_ids = np.full((height, width), -1, dtype=np.int32)
    pixel_ids[candidate] = np.arange(int(candidate.sum()), dtype=np.int32)
    n = int(candidate.sum())
    if n == 0:
        return np.zeros_like(candidate), np.zeros((height, width), dtype=np.int32)
    rows, columns = [np.arange(n, dtype=np.int32)], [np.arange(n, dtype=np.int32)]
    for channel, (dy, dx) in enumerate(OFFSETS):
        ya, yb = max(0, -dy), min(height, height - dy)
        xa, xb = max(0, -dx), min(width, width - dx)
        source = pixel_ids[ya:yb, xa:xb]
        destination = pixel_ids[ya + dy : yb + dy, xa + dx : xb + dx]
        threshold = max(edge_threshold, 0.8) if channel >= 4 else edge_threshold
        allowed = (
            (source >= 0) & (destination >= 0) & (relation[channel, ya:yb, xa:xb] >= threshold)
        )
        if not allowed.any():
            continue
        first_z = z[:, ya:yb, xa:xb]
        last_z = z[:, ya + dy : yb + dy, xa + dx : xb + dx]
        similarity = (first_z * last_z).sum(axis=0)
        allowed &= similarity >= similarity_threshold
        if max(abs(dx), abs(dy)) > 1:
            steps = max(abs(dx), abs(dy))
            for step in range(1, steps):
                middle = pixel_ids[
                    ya + step * dy // steps : yb + step * dy // steps,
                    xa + step * dx // steps : xb + step * dx // steps,
                ]
                allowed &= middle >= 0
        if allowed.any():
            rows.append(source[allowed])
            columns.append(destination[allowed])
    graph = coo_matrix(
        (
            np.ones(sum(len(v) for v in rows), dtype=np.uint8),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(n, n),
    ).tocsr()
    count, component = connected_components(graph, directed=False)
    sizes = np.bincount(component, minlength=count)
    seeds = np.bincount(component[region[candidate] >= seed_threshold], minlength=count)
    keep = (sizes >= min_area) & (seeds > 0)
    mapping = np.zeros(count, dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labels = np.zeros((height, width), dtype=np.int32)
    labels[candidate] = mapping[component]
    return labels > 0, labels
