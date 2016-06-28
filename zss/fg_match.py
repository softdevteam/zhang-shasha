import heapq, collections

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

def match_nodes_by_fingerprint_for_collapse(A, B, fingerprint_fn, min_depth=0):
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



def jaccard_similarity_size(size_a, size_b):
    return float(min(size_a, size_b)) / (float(max(size_a, size_b)) + 1.0e-9)


class FeatureVector (object):
    def __init__(self, xs):
        self.v = xs
        self.sum = sum(self.v)

    @staticmethod
    def zeros(N):
        return FeatureVector([0 for i in xrange(N)])

    def copy(self):
        return FeatureVector(list(self.v))

    def jaccard_similarity(self, b):
        intersection = 0.0
        union = 0.0
        n = min(len(self.v), len(b.v))
        for x, y in zip(self.v, b.v):
            intersection += min(x, y)
            union += max(x, y)
        for x in self.v[n:]:
            union += x
        for y in b.v[n:]:
            union += y
        return intersection / (union + 1.0e-9)

    def jaccard_similarity_upper_bound(self, b):
        return jaccard_similarity_size(self.sum, b.sum)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]

    def __setitem__(self, key, value):
        diff = value - self.v[key]
        self.v[key] = value
        self.sum += diff

    def __add__(self, other):
        if isinstance(other, FeatureVector):
            xs = [a+b for a,b in zip(self.v, other.v)]
            if len(self.v) > len(other.v):
                return FeatureVector(xs + self.v[len(xs):])
            elif len(other.v) > len(self.v):
                return FeatureVector(xs + other.v[len(xs):])
            else:
                return FeatureVector(xs)
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, FeatureVector):
            if len(self.v) >= len(other.v):
                for i, b in enumerate(other.v):
                    self.v[i] += other.v[i]
            else:
                xs = [a+b for a,b in zip(self.v, other.v)]
                self.v = xs + other.v[len(xs):]
            self.sum += other.sum
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, FeatureVector):
            xs = [a-b for a,b in zip(self.v, other.v)]
            if len(self.v) > len(other.v):
                return FeatureVector(xs + self.v[len(xs):])
            elif len(other.v) > len(self.v):
                return FeatureVector(xs + [-x for x in other.v[len(xs):]])
            else:
                return FeatureVector(xs)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, FeatureVector):
            return FeatureVector([a*b for a,b in zip(self.v, other.v)])
        elif isinstance(other, (float, int, long)):
            return FeatureVector([x*other for x in self.v])
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, FeatureVector):
            return FeatureVector([min(a,b) for a,b in zip(self.v, other.v)])
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, FeatureVector):
            xs = [max(a,b) for a,b in zip(self.v, other.v)]
            if len(self.v) > len(other.v):
                return FeatureVector(xs + self.v[len(xs):])
            elif len(other.v) > len(self.v):
                return FeatureVector(xs + other.v[len(xs):])
            else:
                return FeatureVector(xs)
        else:
            return NotImplemented

    def __abs__(self):
        return FeatureVector([abs(x) for x in self.v])

    def __neg__(self):
        return FeatureVector([-x for x in self.v])

    def __eq__(self, other):
        if isinstance(other, FeatureVector):
            return self.v == other.v
        elif isinstance(other, (list, tuple)):
            return self.v == other
        else:
            return NotImplemented

    def __str__(self):
        return 'FeatureVector({})'.format(self.v)

    def __repr__(self):
        return 'FeatureVector({})'.format(self.v)



class FeatureVectorTable (object):
    def __init__(self):
        self.fingerprints = {}
        self.fg_feature_vectors = []

    @property
    def n_fingerprints(self):
        return len(self.fingerprints)


    def add_tree(self, tree):
        nodes_by_index = {}
        tree.update_fingerprint_index(self.fingerprints, nodes_by_index, None)

        self.fg_feature_vectors.extend([None for i in xrange(len(self.fg_feature_vectors), len(self.fingerprints))])

        self._build_tree_feature_vectors_bottom_up(tree)
        self._build_node_feature_vectors_top_down(tree)

        return nodes_by_index


    def _build_tree_feature_vectors_bottom_up(self, root):
        n = self.n_fingerprints
        root.left_sibling_feats = FeatureVector.zeros(n)
        root.right_sibling_feats = FeatureVector.zeros(n)
        self._build_node_feature_vectors_bottom_up(root)

    def _build_node_feature_vectors_bottom_up(self, node):
        # Update child feature vectors
        for child in node.children:
            self._build_node_feature_vectors_bottom_up(child)

        # Compute cumulative features of children and set
        if len(node.children) > 0:
            cumulative_child_feats = [FeatureVector.zeros(self.n_fingerprints)]
            for child in node.children:
                cumulative_child_feats.append(cumulative_child_feats[-1] + child.feature_vector)

            for i, child in enumerate(node.children):
                child.left_sibling_feats = cumulative_child_feats[i]
                child.right_sibling_feats = cumulative_child_feats[-1] - cumulative_child_feats[i+1]

        fg = node.fingerprint_index
        feat = self.fg_feature_vectors[fg]
        if feat is None:
            feat = FeatureVector.zeros(self.n_fingerprints)
            feat[fg] += 1

            if len(node.children) > 0:
                for child in node.children:
                    feat += child.feature_vector

        node.feature_vector = feat


    def _build_node_feature_vectors_top_down(self, node, fg_left=None, fg_right=None):
        if fg_left is None and fg_right is None:
            n = self.n_fingerprints
            fg_left = FeatureVector.zeros(n)
            fg_right = FeatureVector.zeros(n)

        node.left_tree_feats = fg_left
        node.right_tree_feats = fg_right

        for child in node.children:
            self._build_node_feature_vectors_top_down(child, fg_left + child.left_sibling_feats,
                                                      fg_right + child.right_sibling_feats)


def score_match_context(a, b):
    return a.left_tree_feats.jaccard_similarity(b.left_tree_feats) + \
        a.right_tree_feats.jaccard_similarity(b.right_tree_feats) + \
        a.left_sibling_feats.jaccard_similarity(b.left_sibling_feats) * 10 + \
        a.right_sibling_feats.jaccard_similarity(b.right_sibling_feats) * 10


def score_match(a, b):
    score = score_match_context(a, b)
    return score + a.feature_vector.jaccard_similarity(b.feature_vector) * 100

def score_match_upper_bound(a, b):
    return a.left_tree_feats.jaccard_similarity_upper_bound(b.left_tree_feats) + \
           a.right_tree_feats.jaccard_similarity_upper_bound(b.right_tree_feats) + \
           a.left_sibling_feats.jaccard_similarity_upper_bound(b.left_sibling_feats) * 10 + \
           a.right_sibling_feats.jaccard_similarity_upper_bound(b.right_sibling_feats) * 10 + \
           a.feature_vector.jaccard_similarity_upper_bound(b.feature_vector) * 100

def score_match_upper_bound_approx(node, subtree_size):
    return 2 + 20 + jaccard_similarity_size(node.feature_vector.sum, subtree_size) * 100


def top_down_match_nodes_by_fingerprint(A, B, min_depth=0):
    matches = []

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
            a_nodes, b_nodes = nodes_by_fg.setdefault(a.fingerprint_index, (list(), list()))
            a_nodes.append(a)
        for b in xs_b:
            a_nodes, b_nodes = nodes_by_fg.setdefault(b.fingerprint_index, (list(), list()))
            b_nodes.append(b)

        for fg, (a_nodes, b_nodes) in nodes_by_fg.items():
            if len(a_nodes) == 1 and len(b_nodes) == 1:
                matches.append((a_nodes[0], b_nodes[0]))
                a_nodes[0].matched = True
                b_nodes[0].matched = True
            elif len(a_nodes) > 0 and len(b_nodes) > 0:
                potential_matches = []
                for a in a_nodes:
                    for b in b_nodes:
                        potential_matches.append((a, b))
                potential_matches.sort(key=lambda pair: score_match_context(pair[0], pair[1]))
                for a, b in potential_matches:
                    if not a.matched and not b.matched:
                        a.matched = True
                        b.matched = True
                        matches.append((a, b))
                for a in a_nodes:
                    if not a.matched:
                        depth_q_A.push_nodes(a.children)
                for b in b_nodes:
                    if not b.matched:
                        depth_q_B.push_nodes(b.children)
            else:
                for a in a_nodes:
                    depth_q_A.push_nodes(a.children)
                for b in b_nodes:
                    depth_q_B.push_nodes(b.children)

    return matches



def apply_bottom_up_match(a, b):
    score = score_match(a, b)
    if score > a.best_match_score and score > b.best_match_score:
        a.best_match = b
        b.best_match = a
        a.best_match_score = b.best_match_score = score


def bottom_up_match_nodes_by_fingerprint(A, B, matches=None):
    if matches is None:
        matches = []

    a_nodes = [a for a in A.iter_unmatched() if not a.matched]
    b_nodes = [b for b in B.iter_unmatched() if not b.matched]

    # print 'BOTTOM UP {} <-> {}'.format(len(a_nodes), len(b_nodes))

    scored_matches = []
    for a in a_nodes:
        for b in b_nodes:
            scored_match = score_match(a, b), a, b
            scored_matches.append(scored_match)

    scored_matches.sort(key=lambda x: x[0], reverse=True)

    for score, a, b in scored_matches:
        if not a.matched and not b.matched:
            matches.append((a, b))
            a.matched = b.matched = True

    return matches

def get_candidate_node(a_node, B):
    b_nodes = B.post_order_unmatched()

    best_score = -1.0
    best_node = None
    for b in b_nodes:
        if not b.matched and not b.is_leaf():
            score = score_match(a_node, b)
            if score > best_score:
                best_score = score
                best_node = b
    return best_node, best_score


def greedy_bottom_up_match_nodes_by_fingerprint(A, B, score_thresh=0.3):
    matches = []

    a_nodes = A.post_order_unmatched()

    # print 'BOTTOM UP {} <-> {}'.format(len(a_nodes), len(b_nodes))

    for a in a_nodes:
        if not a.matched and not a.is_leaf():
            c, score = get_candidate_node(a, B)
            if score >= score_thresh:
                bottom_up_match_nodes_by_fingerprint(a, c, matches)

    return matches





import unittest

class Test_FeatureVector (unittest.TestCase):
    def test_get_set(self):
        xs = FeatureVector([1,2,3])
        ys = FeatureVector([1,2,3])
        self.assertEqual(xs.v, [1,2,3])
        self.assertEqual(xs.sum, 6)
        self.assertEqual(len(xs), 3)
        self.assertEqual(xs[0], 1)
        self.assertEqual(xs[1], 2)
        self.assertEqual(xs[2], 3)
        self.assertEqual(xs, ys)
        self.assertEqual(xs.copy(), xs)


    def test_arithmetic(self):
        xs = FeatureVector([1,2,3])
        ys = FeatureVector([10, 20, 30, 40, 50, 60])

        xs_iadd_ys = xs.copy()
        xs_iadd_ys += ys
        ys_iadd_xs = ys.copy()
        ys_iadd_xs += xs

        self.assertEqual(xs + ys, [11, 22, 33, 40, 50, 60])
        self.assertEqual(ys + xs, [11, 22, 33, 40, 50, 60])
        self.assertEqual(xs_iadd_ys, [11, 22, 33, 40, 50, 60])
        self.assertEqual(ys_iadd_xs, [11, 22, 33, 40, 50, 60])
        self.assertEqual(xs - ys, [-9, -18, -27, -40, -50, -60])
        self.assertEqual(ys - xs, [9, 18, 27, 40, 50, 60])
        self.assertEqual(xs * ys, [10, 40, 90])
        self.assertEqual(ys * xs, [10, 40, 90])
        self.assertEqual(xs * 10, [10, 20, 30])
        self.assertEqual(ys * 10, [100, 200, 300, 400, 500, 600])
        self.assertEqual(xs & ys, [1, 2, 3])
        self.assertEqual(ys & xs, [1, 2, 3])
        self.assertEqual(xs | ys, [10, 20, 30, 40, 50, 60])
        self.assertEqual(ys | xs, [10, 20, 30, 40, 50, 60])

