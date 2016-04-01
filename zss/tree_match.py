import time
import simple_tree, compare, zs_memo, fgcompact


def compute_shape_fingerprints(A, B, A_ignore=None, B_ignore=None):
    fingerprints = {}
    A_nodes_by_index = {}
    B_nodes_by_index = {}
    A.update_fingerprint_index(fingerprints, A_nodes_by_index, A_ignore)
    B.update_fingerprint_index(fingerprints, B_nodes_by_index, B_ignore)
    return fingerprints, A_nodes_by_index, B_nodes_by_index


def compute_content_fingerprints(A, B):
    fingerprints = {}
    A_nodes_by_index = {}
    B_nodes_by_index = {}
    A.update_content_fingerprint_index(fingerprints, A_nodes_by_index)
    B.update_content_fingerprint_index(fingerprints, B_nodes_by_index)
    return fingerprints, A_nodes_by_index, B_nodes_by_index


def tree_match(A, B, verbose=False, distance_fn=None):
    if distance_fn is None:
        distance_fn = zs_memo.simple_distance

    A_opt = A
    B_opt = B

    comparison_permitted_by_label = None

    A_nodes_to_collapse = set()
    B_nodes_to_collapse = set()

    base_content_fingerprints, A_nodes_by_content_index, B_nodes_by_content_index = compute_content_fingerprints(A_opt, B_opt)
    for A_fg_index, A_nodes in A_nodes_by_content_index.items():
        B_nodes = B_nodes_by_content_index.get(A_fg_index)
        if B_nodes is not None:
            if len(A_nodes) == len(B_nodes):
                A_nodes_to_collapse = A_nodes_to_collapse.union(A_nodes)
                B_nodes_to_collapse = B_nodes_to_collapse.union(B_nodes)

    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    A_nodes_to_collapse = set()
    B_nodes_to_collapse = set()

    base_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)
    for A_fg_index, A_nodes in A_nodes_by_index.items():
        B_nodes = B_nodes_by_index.get(A_fg_index)
        if B_nodes is not None:
            if len(A_nodes) == len(B_nodes):
                A_nodes_to_collapse = A_nodes_to_collapse.union(A_nodes)
                B_nodes_to_collapse = B_nodes_to_collapse.union(B_nodes)


    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    opt_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)

    if verbose:
        print 'BASE CASE TREE STATS: |A|={0}, |B|={1}, A.height={2}, B.height={3}, |content_fingerprints(A, B)|={4}, |shape_fingerprints(A, B)|={5}'.format(
            len([x for x in A.iter()]), len([x for x in B.iter()]),
            A.depth, B.depth, len(base_content_fingerprints), len(base_fingerprints))
        print 'OPTIMISED TREE STATS: |A_opt|={0}, |B_opt|={1}, A_opt.height={2}, B_opt.height={3}, ' \
            '|shape_fingerprints(A_opt, B_opt)|={4}'.format(
            len([x for x in A_opt.iter()]), len([x for x in B_opt.iter()]),
            A_opt.depth, B_opt.depth, len(opt_fingerprints))

    t1 = time.time()
    d, node_matches = distance_fn(A_opt, B_opt, len(opt_fingerprints), simple_tree.Node.get_children, simple_tree.Node.get_label,
                                comparison_filter=comparison_permitted_by_label, verbose=verbose)
    t2 = time.time()

    full_matches = fgcompact.compacted_match_list_to_node_list(node_matches)

    return d, full_matches, t2 - t1
