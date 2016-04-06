import time
import simple_tree, compare, zs_memo, fgcompact, fg_match



def tree_match(A, B, verbose=False, distance_fn=None):
    if distance_fn is None:
        distance_fn = zs_memo.simple_distance

    A_opt = A
    B_opt = B

    comparison_permitted_by_label = None


    t1 = time.time()

    base_content_fingerprints, A_nodes_by_content_index, B_nodes_by_content_index = fg_match.compute_content_fingerprints(A_opt, B_opt)
    A_nodes_to_collapse, B_nodes_to_collapse = fg_match.match_nodes_by_fingerprint_for_collapse(A_opt, B_opt, lambda node: node.content_fingerprint_index)
    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    base_fingerprints, A_nodes_by_index, B_nodes_by_index = fg_match.compute_shape_fingerprints(A_opt, B_opt)
    A_nodes_to_collapse, B_nodes_to_collapse = fg_match.match_nodes_by_fingerprint_for_collapse(A_opt, B_opt, lambda node: node.fingerprint_index)
    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    t2 = time.time()

    opt_fingerprints, A_nodes_by_index, B_nodes_by_index = fg_match.compute_shape_fingerprints(A_opt, B_opt)

    if verbose:
        print 'BASE CASE TREE STATS: |A|={0}, |B|={1}, A.height={2}, B.height={3}, |content_fingerprints(A, B)|={4}, |shape_fingerprints(A, B)|={5}'.format(
            len([x for x in A.iter()]), len([x for x in B.iter()]),
            A.depth, B.depth, len(base_content_fingerprints), len(base_fingerprints))
        print 'OPTIMISED TREE STATS: |A_opt|={0}, |B_opt|={1}, A_opt.height={2}, B_opt.height={3}, ' \
            '|shape_fingerprints(A_opt, B_opt)|={4}, opt time={5:.2f}s'.format(
            len([x for x in A_opt.iter()]), len([x for x in B_opt.iter()]),
            A_opt.depth, B_opt.depth, len(opt_fingerprints), t2-t1)

    t1 = time.time()
    d, node_matches = distance_fn(A_opt, B_opt, len(opt_fingerprints), simple_tree.Node.get_children, simple_tree.Node.get_label,
                                comparison_filter=comparison_permitted_by_label, verbose=verbose)
    t2 = time.time()

    full_matches = fgcompact.compacted_match_list_to_node_list(node_matches)

    return d, full_matches, t2 - t1
