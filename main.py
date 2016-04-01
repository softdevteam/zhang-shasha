import datetime, collections, sys, argparse

from zss import simple_tree, compare, zs_memo, fgcompact, edit_script

import parser, ast, _ast

import type_pruning

from ast_to_simple_tree import ASTConverter
from source_text import SourceText, Marker, longest_common_prefix, longest_common_suffix
import tree_flattening


codea1 = """
def gauss(x, a, b, c):
    return a * np.exp(-(x-b**2)/2*c**2)
"""

codea2 = """
def gauss(a, b, c):
    return a * np.exp(-(x-b)**2/2*c**2)
"""

codeb1 = """
def timed_call(f, x, y, z, N=50000):
    t1 = time.time()

    for i in range(N):
       f(x, y, z)

    t2 = time.time()

    return t2 - t1
"""

codeb2 = """
def timed_call(f, x, y, z, q, N=50000):
    t1 = time.time()

    for i in range(N):
       f(x, y, z, q)

    t2 = time.time()

    return t2 - t1
"""

codec1 = """
def gauss(x, a, b, c):
    return a * np.exp(-(x-b**2)/2*c**2+f(x**a(b**c(**x))))
"""

codec2 = """
def gauss(a, b, c):
    return a * np.exp(-(x-b)**2/2*c**2+f(x**a(b**c(**x))))
"""

coded1 = """
def tuple_to_tree(x):
    label = x[0]
    child_node_count = len([y for y in x[1:] if isinstance(y, tuple)])
    if child_node_count == len(x)-1:
        return simple_tree.Node(str(label), [tuple_to_tree(y) for y in x[1:]])
    else:
        label = ':'.join([str(y) for y in x])
        return simple_tree.Node(label)
"""

coded2 = """
def tuple_to_tree(x):
    label = x[0]
    return simple_tree.Node(str(label), [tuple_to_tree(y) for y in x[1:] if isinstance(y, tuple)])
"""

codee1 = """
def simple_distance(A, B, get_children=Node.get_children,
        get_label=Node.get_label, label_dist=strdist):
    counter = itertools.count()

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

    for a in all_A:
        for b in all_B:
            total_node_pairings += 1
            compare_key = (a.sha, b.sha)
            child_compares = len(a.children) * len(b.children)

            total_node_compares += child_compares
            if compare_key not in unique_node_pairings:
                pruned_node_compares += child_compares

            unique_node_pairings.add((a.sha, b.sha))

    print('A: {0} nodes, {1} unique'.format(A.subtree_size(), len(a_sha_to_index)))
    print('B: {0} nodes, {1} unique'.format(B.subtree_size(), len(b_sha_to_index)))
    print('intersection: {0} nodes'.format(len(intersection)))
    print('node pairings: {0} total, {1} unique'.format(total_node_pairings, len(unique_node_pairings)))
    print('node comparisons: {0} total, {1} unique'.format(total_node_compares, pruned_node_compares))

    return distance(
        A, B, get_children,
        insert_cost=lambda node: label_dist('', get_label(node)),
        remove_cost=lambda node: label_dist(get_label(node), ''),
        update_cost=lambda a, b: label_dist(get_label(a), get_label(b)),
    )
"""

codee2 = """
def simple_distance(A, B, get_children=Node.get_children,
        get_label=Node.get_label, label_dist=strdist):
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

    for a in all_A:
        for b in all_B:
            total_node_pairings += 1
            compare_key = (a.sha, b.sha)
            child_compares = len(a.children) * len(b.children)

            total_node_compares += child_compares
            if compare_key not in unique_node_pairings:
                pruned_node_compares += child_compares

            unique_node_pairings.add((a.sha, b.sha))

    print('A: {0} nodes, {1} unique'.format(A.subtree_size(), len(a_sha_to_index)))
    print('B: {0} nodes, {1} unique'.format(B.subtree_size(), len(b_sha_to_index)))
    print('intersection: {0} nodes'.format(len(intersection)))
    print('node pairings: {0} total, {1} unique'.format(total_node_pairings, len(unique_node_pairings)))
    print('node comparisons: {0} total, {1} unique'.format(total_node_compares, pruned_node_compares))

    all_A.sort(key=lambda x: x.depth)
    all_B.sort(key=lambda x: x.depth)

    return distance(
        A, B, get_children,
        insert_cost=lambda node: label_dist('', get_label(node)),
        remove_cost=lambda node: label_dist(get_label(node), ''),
        update_cost=lambda a, b: label_dist(get_label(a), get_label(b)),
    )
"""



def common_prefix_and_suffix_matches(A_src, A, B_src, B):
    A_prefix_end, B_prefix_end = A_src.markers_at_end_of_longest_common_prefix(B_src)
    A_suffix_start, B_suffix_start = A_src.markers_at_start_of_longest_common_suffix(B_src)
    matches = []
    A.common_prefix_matches(matches, B, A_prefix_end, B_prefix_end)
    A.common_suffix_matches(matches, B, A_suffix_start, B_suffix_start)
    return matches


def prune_prefix_and_suffix_matches(A_src, A, B_src, B):
    A_prefix_end, B_prefix_end = A_src.markers_at_end_of_longest_common_prefix(B_src)
    A_suffix_start, B_suffix_start = A_src.markers_at_start_of_longest_common_suffix(B_src)
    A_pruned = A.prune(A_prefix_end, A_suffix_start)
    B_pruned = B.prune(B_prefix_end, B_suffix_start)
    return A_pruned, B_pruned


def flatten_trees(A, B):
    flattened_types, retained_types = tree_flattening.flatten_types()
    flattened_type_names = [t.__name__ for t in flattened_types]

    def flatten_pred_fn(x):
        return x.label in flattened_type_names

    A_flat = A.flatten(flatten_pred_fn)
    B_flat = B.flatten(flatten_pred_fn)

    return A_flat, B_flat


def compute_shape_fingerprints(A, B):
    fingerprints = {}
    A_nodes_by_index = {}
    B_nodes_by_index = {}
    A.update_fingerprint_index(fingerprints, A_nodes_by_index)
    B.update_fingerprint_index(fingerprints, B_nodes_by_index)
    return fingerprints, A_nodes_by_index, B_nodes_by_index



def test(A_src, B_src, type_filtering=False, flatten=False, common_prefix_suffix=False,
         memo=False, fingerprint_matching=False, fingerprint_compaction=False, repeats=1):
    conv = ASTConverter()


    A = conv.parse(A_src)
    B = conv.parse(B_src)


    node_classes = [getattr(_ast, name) for name in dir(_ast)]
    node_classes = [cls for cls in node_classes if isinstance(cls, type) and issubclass(cls, _ast.AST)]
    node_classes.sort(key=lambda x: x.__name__)


    A_opt = A
    B_opt = B


    if type_filtering:
        comparison_permitted = type_pruning.compute_node_type_compatibility_by_grammar(node_classes)

        comparison_permitted_by_label = {
            (a.__name__, b.__name__): v for ((a,b), v) in comparison_permitted.items()
        }
    else:
        comparison_permitted_by_label = None


    if flatten:
        A_opt, B_opt = flatten_trees(A_opt, B_opt)

    if common_prefix_suffix:
        A_opt, B_opt = prune_prefix_and_suffix_matches(A_src, A_opt, B_src, B_opt)


    base_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)


    unique_node_matches = None
    potential_match_fingerprints = None
    num_potentially_matching_nodes_A = None
    num_potentially_matching_nodes_B = None

    if fingerprint_matching:
        unique_node_matches = []
        potential_match_fingerprints = set()
        num_potentially_matching_nodes_A = 0
        num_potentially_matching_nodes_B = 0
        for A_fg_index, A_nodes in A_nodes_by_index.items():
            B_nodes = B_nodes_by_index.get(A_fg_index)
            if B_nodes is not None:
                if len(A_nodes) == 1 and len(B_nodes) == 1:
                    unique_node_matches.append((A_nodes[0], B_nodes[0]))
                elif len(A_nodes) == len(B_nodes):
                    potential_match_fingerprints.add(A_fg_index)
                    num_potentially_matching_nodes_A += len(A_nodes)
                    num_potentially_matching_nodes_B += len(B_nodes)
    elif fingerprint_compaction:
        A_nodes_to_collapse = set()
        B_nodes_to_collapse = set()
        for A_fg_index, A_nodes in A_nodes_by_index.items():
            B_nodes = B_nodes_by_index.get(A_fg_index)
            if B_nodes is not None:
                if len(A_nodes) == len(B_nodes):
                    A_nodes_to_collapse = A_nodes_to_collapse.union(A_nodes)
                    B_nodes_to_collapse = B_nodes_to_collapse.union(B_nodes)
        A_opt = A_opt.compact(A_nodes_to_collapse)
        B_opt = B_opt.compact(B_nodes_to_collapse)



    opt_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)


    print 'SOURCE CODE STATS: |A_src|={0}, |B_src|={1}, |A_src.lines|={2}, |B_src.lines|={3}, ' \
          '|common prefix lines|={4}, |common suffix lines|={5}'.format(
        len(A_src), len(B_src), len(A_src.lines), len(B_src.lines), longest_common_prefix(A_src.lines, B_src.lines),
        longest_common_suffix(A_src.lines, B_src.lines))
    print 'BASE CASE TREE STATS: |A|={0}, |B|={1}, A.height={2}, B.height={3}, |shape_fingerprints(A, B)|={4}'.format(
        len([x for x in A.iter()]), len([x for x in B.iter()]),
        A.depth, B.depth, len(base_fingerprints))
    print 'OPTIMISED TREE STATS: |A_opt|={0}, |B_opt|={1}, A_opt.height={2}, B_opt.height={3}, ' \
        '|shape_fingerprints(A_opt, B_opt)|={4}'.format(
        len([x for x in A_opt.iter()]), len([x for x in B_opt.iter()]),
        A_opt.depth, B_opt.depth, len(opt_fingerprints))
    if unique_node_matches is not None:
        print '|unique fingerprint matches|={0}, |potential match fgs|={1}, ' \
              '|potential matches in A|={2}, |potential matches in B|={3}'.format(
                    len(unique_node_matches), len(potential_match_fingerprints),
                    num_potentially_matching_nodes_A, num_potentially_matching_nodes_B)


    print ''
    min_dt = None
    max_dt = None
    sum_dt = None
    for i in xrange(repeats):
        t1 = datetime.datetime.now()
        if memo:
            d, node_matches = zs_memo.simple_distance(A_opt, B_opt, len(opt_fingerprints), simple_tree.Node.get_children, simple_tree.Node.get_label,
                                        comparison_filter=comparison_permitted_by_label,
                                        unique_match_constraints=unique_node_matches,
                                        potential_match_fingerprints=potential_match_fingerprints, verbose=True)
        else:
            d, node_matches = compare.simple_distance(A_opt, B_opt, len(opt_fingerprints), simple_tree.Node.get_children, simple_tree.Node.get_label,
                                        comparison_filter=comparison_permitted_by_label,
                                        unique_match_constraints=unique_node_matches,
                                        potential_match_fingerprints=potential_match_fingerprints, verbose=True)
        t2 = datetime.datetime.now()
        compare.check_match_list(node_matches)
        dt = t2 - t1
        min_dt = min(dt, min_dt) if min_dt is not None else dt
        max_dt = max(dt, max_dt) if max_dt is not None else dt
        sum_dt = sum_dt + dt if sum_dt is not None else dt

    node_matches_full = fgcompact.compacted_match_list_to_node_list(node_matches)


    diffs = edit_script.edit_script(A, B, node_matches_full)
    X = A.clone()
    A_ids = {a.merge_id for a in A.iter()}
    X_ids = {x.merge_id for x in X.iter()}
    assert A_ids == X_ids
    merge_id_to_node = {}
    for node in X.iter():
        merge_id_to_node[node.merge_id] = node
    for diff in diffs:
        diff.apply(merge_id_to_node)

    assert X == B

    print 'Distance={}, |node matches|={}, |full_matches|={}, |diffs|={}, took min {} max {} avg {}'.format(d, len(node_matches), len(node_matches_full), len(diffs), min_dt, max_dt, sum_dt / repeats)




def get_data(data):
    if data == 'a':
        return SourceText(codea1), SourceText(codea2)
    elif data == 'b':
        return SourceText(codeb1), SourceText(codeb2)
    elif data == 'c':
        return SourceText(codec1), SourceText(codec2)
    elif data == 'd':
        return SourceText(coded1), SourceText(coded2)
    elif data == 'e':
        return SourceText(codee1), SourceText(codee2)
    elif data == 'exa':
        return SourceText.from_file(open('example_test_v1.py', 'r')),\
               SourceText.from_file(open('example_test_v2.py', 'r'))
    elif data == 'exb':
        return SourceText.from_file(open('example_test_b_v1.py', 'r')),\
               SourceText.from_file(open('example_test_b_v2.py', 'r'))
    elif data == 'exb2':
        return SourceText.from_file(open('example_test_b_v1.py', 'r')),\
               SourceText.from_file(open('example_test_b_v3.py', 'r'))
    elif data == 'exab':
        return SourceText.from_file(open('example_test_v1.py', 'r')),\
               SourceText.from_file(open('example_test_b_v2.py', 'r'))
    else:
        raise ValueError("Could not get example named {0}".format(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help="Which data to process; 'a'-'e' for small examples, "
                        "'exa' for a ~360 line Python file, 'exb' for a ~1600 line Python file with localised changes, "
                        "'exb2' for a 'exb' with changes at the beginning and end")
    parser.add_argument('--type_filter', action='store_true', help="Enable type compatibility filtering")
    parser.add_argument('--flatten', action='store_true', help="Enable tree flattening optimisation")
    parser.add_argument('--common_ends', action='store_true', help="Remove common prefix and suffix")
    parser.add_argument('--memo', action='store_true', help="Uses memoised Zhang-Shasha")
    parser.add_argument('--fg_match', action='store_true', help="Enable fingerprint matching")
    parser.add_argument('--fg_compact', action='store_true', help="Enable fingerprint compaction")
    parser.add_argument('--repeats', type=int, default=1, help="number of repetitions")
    args = parser.parse_args()

    A_src, B_src = get_data(args.data)

    test(A_src, B_src, args.type_filter, args.flatten, args.common_ends, args.memo, args.fg_match,
         args.fg_compact, args.repeats)

