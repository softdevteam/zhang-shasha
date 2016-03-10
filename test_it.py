import datetime, collections, sys

from zss import simple_tree, compare

import parser, ast, _ast

import type_pruning

from ast_to_simple_tree import ASTConverter
from source_text import SourceText, Marker



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




def test():
    conv = ASTConverter()

    # st1 = parser.suite(coded1)
    # st2 = parser.suite(coded2)
    #
    # A = tuple_to_tree(st1.totuple())
    # B = tuple_to_tree(st2.totuple())

    # A = SourceText.from_file(codea1)
    # B = SourceText.from_file(codea2)
    # A = SourceText.from_file(coded1)
    # B = SourceText.from_file(coded2)
    # A = SourceText.from_file(codee1)
    # B = SourceText.from_file(codee2)

    A_src = SourceText.from_file(open('example_test_v1.py', 'r'))
    B_src = SourceText.from_file(open('example_test_v2.py', 'r'))
    # A_src = SourceText.from_file(open('example_test_b_v1.py', 'r'))
    # B_src = SourceText.from_file(open('example_test_b_v2.py', 'r'))

    A_prefix_end, B_prefix_end = A_src.markers_at_end_of_longest_common_prefix(B_src)
    A_suffix_start, B_suffix_start = A_src.markers_at_start_of_longest_common_suffix(B_src)

    A = conv.parse(A_src)
    B = conv.parse(B_src)

    matches = []
    A.common_prefix_matches(matches, B, A_prefix_end, B_prefix_end)
    A.common_suffix_matches(matches, B, A_suffix_start, B_suffix_start)


    node_classes = [getattr(_ast, name) for name in dir(_ast)]
    node_classes = [cls for cls in node_classes if isinstance(cls, type) and issubclass(cls, _ast.AST)]
    node_classes.sort(key=lambda x: x.__name__)


    comparison_permitted = type_pruning.compute_node_type_compatibility_by_grammar(node_classes)

    # print type_pruning.type_compatbility_map_to_matrix(node_classes, comparison_permitted)

    comparison_permitted_by_label = {
        (a.__name__, b.__name__): v for ((a,b), v) in comparison_permitted.items()
    }

    print '|A|={0}, |B|={1}, |matches|={2}'.format(len([x for x in A.iter()]), len([x for x in B.iter()]), len(matches))

    total_comparisons = 0
    filtered_comparisons = 0
    for x in A.iter():
        for y in B.iter():
            key = x.label, y.label
            if comparison_permitted_by_label[key]:
                filtered_comparisons += 1
            total_comparisons += 1

    print '{0} / {1} comparisons passed'.format(filtered_comparisons, total_comparisons)


    t1 = datetime.datetime.now()

    # matches = None
    d = compare.simple_distance(A, B, simple_tree.Node.get_children, simple_tree.Node.get_label,
                                comparison_filter=comparison_permitted_by_label,
                                match_constraints=matches)
    # d = compare.simple_distance(A, B, simple_tree.Node.get_children, simple_tree.Node.get_label, comparison_filter=None)

    t2 = datetime.datetime.now()
    print t2 - t1

    print 'Distance={0}'.format(d)


if __name__ == '__main__':
    test()

