import time
import simple_tree, compare, zs_memo, fgcompact


class DepthNodeQueue (object):
    def __init__(self, root):
        self.max_depth = root.depth
        self._entries_by_depth = [list() for i in range(self.max_depth+1)]
        self._entries_by_depth[root.depth].append(root)

    def push_nodes(self, nodes):
        for node in nodes:
            node_depth = node.depth
            self._entries_by_depth[node_depth].append(node)
            self.max_depth = max(self.max_depth, node_depth)

    def pop_nodes_at_max_depth(self):
        if self.max_depth == 0:
            return []
        else:
            nodes = self._entries_by_depth[self.max_depth]
            self._entries_by_depth[self.max_depth] = []
            while self.max_depth > 0 and len(self._entries_by_depth[self.max_depth]) == 0:
                self.max_depth -= 1
            return nodes





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


USE_HASH_TABLE_THRESHOLD = 512

def match_nodes_by_fingerprint(A, B, fingerprint_fn, min_depth=0):
    A_nodes_to_collapse = set()
    B_nodes_to_collapse = set()

    depth_q_A = DepthNodeQueue(A)
    depth_q_B = DepthNodeQueue(B)

    while depth_q_A.max_depth > min_depth:
        while depth_q_A.max_depth != depth_q_B.max_depth:
            if depth_q_A.max_depth > depth_q_B.max_depth:
                nodes = depth_q_A.pop_nodes_at_max_depth()
                for node in nodes:
                    depth_q_A.push_nodes(node.children)
            else:
                nodes = depth_q_B.pop_nodes_at_max_depth()
                for node in nodes:
                    depth_q_B.push_nodes(node.children)
        if depth_q_A.max_depth <= min_depth:
            break

        xs_a = depth_q_A.pop_nodes_at_max_depth()
        xs_b = depth_q_B.pop_nodes_at_max_depth()

        nodes_by_fg = {}
        for a in xs_a:
            a_nodes, b_nodes = nodes_by_fg.setdefault(fingerprint_fn(a), (list(), list()))
            a_nodes.append(a)
        for b in xs_b:
            a_nodes, b_nodes = nodes_by_fg.setdefault(fingerprint_fn(b), (list(), list()))
            b_nodes.append(b)

        for fg, (a_nodes, b_nodes) in nodes_by_fg.items():
            if len(a_nodes) == len(b_nodes):
                A_nodes_to_collapse = A_nodes_to_collapse.union(a_nodes)
                B_nodes_to_collapse = B_nodes_to_collapse.union(b_nodes)
            else:
                for a in a_nodes:
                    depth_q_A.push_nodes(a.children)
                for b in b_nodes:
                    depth_q_B.push_nodes(b.children)

    return A_nodes_to_collapse, B_nodes_to_collapse



def tree_match(A, B, verbose=False, distance_fn=None):
    if distance_fn is None:
        distance_fn = zs_memo.simple_distance

    A_opt = A
    B_opt = B

    comparison_permitted_by_label = None


    t1 = time.time()

    base_content_fingerprints, A_nodes_by_content_index, B_nodes_by_content_index = compute_content_fingerprints(A_opt, B_opt)
    A_nodes_to_collapse, B_nodes_to_collapse = match_nodes_by_fingerprint(A_opt, B_opt, lambda node: node.content_fingerprint_index)
    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    base_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)
    A_nodes_to_collapse, B_nodes_to_collapse = match_nodes_by_fingerprint(A_opt, B_opt, lambda node: node.fingerprint_index)
    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)

    t2 = time.time()

    opt_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)

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
