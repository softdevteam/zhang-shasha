

def compact_by_fingerprint(A, B, update_fingerprints_fn):
    fingerprints = {}
    A_nodes_by_index = {}
    B_nodes_by_index = {}
    update_fingerprints_fn(fingerprints, A_nodes_by_index, A)
    update_fingerprints_fn(fingerprints, B_nodes_by_index, B)

    for A_fg_index, A_nodes in A_nodes_by_index.items():
        B_nodes = B_nodes_by_index.get(A_fg_index)
        if B_nodes is not None:
            if len(A_nodes) == len(B_nodes):
                A_nodes_to_collapse = A_nodes_to_collapse.union(A_nodes)
                B_nodes_to_collapse = B_nodes_to_collapse.union(B_nodes)

    A_opt = A.compact(A_nodes_to_collapse)
    B_opt = B.compact(B_nodes_to_collapse)

    return A_opt, B_opt, fingerprints, A_nodes_by_index, B_nodes_by_index


def match_list_to_node_list(A, A_nodes, B, B_nodes, match_list):
    return [(A_nodes[i], B_nodes[j]) for i, j in match_list]


def compacted_match_list_to_node_list(match_list):
    node_matches = []
    for a_i, b_j in match_list:
        if a_i.compacted and b_j.compacted:
            # Check that the fingerprints match
            if a_i.fingerprint_index == b_j.fingerprint_index:
                # Shape fingerprints match; iterate over the subtrees rooted at a_i and b_j
                for x, y in zip(a_i.original_node.iter(), b_j.original_node.iter()):
                    node_matches.append((x, y))
            # If the fingerprints don't match we cannot match the nodes within
        elif not a_i.compacted and not b_j.compacted:
            # Not compacted; add the original nodes as matches
            node_matches.append((a_i.original_node, b_j.original_node))

    return node_matches

