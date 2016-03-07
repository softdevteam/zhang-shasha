try:
    import numpy as np
    zeros = np.zeros
except ImportError:
    def py_zeros(dim, pytype):
        assert len(dim) == 2
        return [[pytype() for y in xrange(dim[1])]
                for x in xrange(dim[0])]
    zeros = py_zeros

try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1


def memoised_compare(A, B):
    a_sha_to_index = {}
    A.update_fingerprint_index(a_sha_to_index)
    b_sha_to_index = {}
    B.update_fingerprint_index(b_sha_to_index)
    intersection = set(a_sha_to_index.keys()).intersection(set(b_sha_to_index.keys()))
    all_A = []
    all_B = []
    A.all_nodes(all_A)
    B.all_nodes(all_B)
    total_node_pairings = 0
    unique_node_pairings = set()
    total_node_compares = 0
    pruned_node_compares = 0

    edit_dist_by_pair = {}

    def node_edit_dist(a, b):
        key = a.fingerprint_index, b.fingerprint_index
        if key in edit_dist_by_pair:
            return edit_dist_by_pair[key]
        else:
            n_a = len(a.children)
            n_b = len(b.children)
            label_cost = 0 if a.label == b.label else 1
            if n_a == 0 and n_b == 0:
                return label_cost
            elif n_a == 0 and n_b > 0:
                return label_cost + sum([node.subtree_size for node in b.children])
            elif n_a > 0 and n_b == 0:
                return label_cost + sum([node.subtree_size for node in a.children])
            else:
                dist_matrix = zeros((n_a + 1, n_b + 1), int)
                for i in xrange(1, n_a+1):
                    dist_matrix[i][0] = dist_matrix[i-1][0] + a.children[i-1].subtree_size
                for j in xrange(1, n_b+1):
                    dist_matrix[0][j] = dist_matrix[0][j-1] + b.children[j-1].subtree_size
                for i in xrange(1, n_a+1):
                    for j in xrange(1, n_b+1):
                        del_cost = dist_matrix[i-1][j] + a.children[i-1].subtree_size
                        ins_cost = dist_matrix[i][j-1] + b.children[j-1].subtree_size
                        upd_cost = dist_matrix[i-1][j-1] + node_edit_dist(a.children[i-1], b.children[j-1])
                        dist_matrix[i][j] = min([del_cost, ins_cost, upd_cost])
            result = label_cost + dist_matrix[n_a][n_b]
            edit_dist_by_pair[key] = result
            return result

    for a in all_A:
        for b in all_B:
            total_node_pairings += 1
            compare_key = (a.fingerprint_index, b.fingerprint_index)
            child_compares = len(a.children) * len(b.children)

            total_node_compares += child_compares
            if compare_key not in unique_node_pairings:
                pruned_node_compares += child_compares

            unique_node_pairings.add(compare_key)

    print('A: {0} nodes, {1} unique'.format(A.subtree_size, len(a_sha_to_index)))
    print('B: {0} nodes, {1} unique'.format(B.subtree_size, len(b_sha_to_index)))
    print('intersection: {0} nodes'.format(len(intersection)))
    print('node pairings: {0} total, {1} unique'.format(total_node_pairings, len(unique_node_pairings)))
    print('node comparisons: {0} total, {1} unique'.format(total_node_compares, pruned_node_compares))
    print('memoised dist: {0}'.format(node_edit_dist(A, B)))

