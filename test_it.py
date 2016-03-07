import datetime, collections

from zss import simple_tree, compare

import parser, ast, _ast


# def tuple_to_tree(x):
#     label = x[0]
#     child_node_count = len([y for y in x[1:] if isinstance(y, tuple)])
#     if child_node_count == len(x)-1:
#         return simple_tree.Node(str(label), [tuple_to_tree(y) for y in x[1:]])
#     else:
#         label = ':'.join([str(y) for y in x])
#         return simple_tree.Node(label)

def tuple_to_tree(x):
    label = x[0]
    return simple_tree.Node(str(label), [tuple_to_tree(y) for y in x[1:] if isinstance(y, tuple)])


class NodeClass (object):
    def __init__(self, cls_name):
        self.cls_name = cls_name

    def node(self, value, children):
        return simple_tree.Node(self.cls_name, children)


class ASTConverter (object):
    """
    Python AST to Gumtree tree converter.
    """
    def __init__(self):
        self._node_classes = {}

    def _get_node_class(self, ast_node):
        t = type(ast_node)
        try:
            return self._node_classes[t]
        except KeyError:
            cls = NodeClass(t.__name__)
            self._node_classes[t] = cls
            return cls

    def _handle_ast_value(self, children, values, x):
        if isinstance(x, ast.AST):
            children.append(self._ast_to_tree(x))
        elif isinstance(x, (list, tuple)):
            for v in x:
                self._handle_ast_value(children, values, v)
        else:
            # raise TypeError('Value type {0}'.format(type(x)))
            pass


    def _ast_to_tree(self, ast_node):
        children = []
        values = []
        for field_name in ast_node._fields:
            field_val = getattr(ast_node, field_name)
            self._handle_ast_value(children, values, field_val)
        node_class = self._get_node_class(ast_node)
        value = '_'.join([str(x) for x in values])
        return node_class.node(value, children)

    def parse(self, code):
        a = ast.parse(code)
        t = self._ast_to_tree(a)
        return t

    def node_classes(self):
        return self._node_classes.keys()



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


def ast_base(cls):
    return cls.__bases__[0]


def test():
    conv = ASTConverter()

    # st1 = parser.suite(coded1)
    # st2 = parser.suite(coded2)
    #
    # A = tuple_to_tree(st1.totuple())
    # B = tuple_to_tree(st2.totuple())

    # A = conv.parse(codea1)
    # B = conv.parse(codea2)
    # A = conv.parse(coded1)
    # B = conv.parse(coded2)
    # A = conv.parse(codee1)
    # B = conv.parse(codee2)

    # A = conv.parse(open('example_test_v1.py', 'r').read())
    # B = conv.parse(open('example_test_v2.py', 'r').read())
    A = conv.parse(open('example_test_b_v1.py', 'r').read())
    B = conv.parse(open('example_test_b_v2.py', 'r').read())

    node_classes = [getattr(_ast, name) for name in dir(_ast)]
    node_classes = [cls for cls in node_classes if isinstance(cls, type) and issubclass(cls, _ast.AST)]
    node_classes.sort(key=lambda x: x.__name__)


    special_classes = {_ast.FunctionDef, _ast.ClassDef}

    comparison_permitted = {}

    for i in xrange(len(node_classes)):
        c_a = node_classes[i]
        for j in xrange(i+1):
            c_b = node_classes[j]
            key1 = c_a, c_b
            key2 = c_b, c_a
            # if c_a in special_classes or c_b in special_classes:
            #     permitted = c_a is c_b
            # elif ast_base(c_a) is ast_base(c_b):
            #     permitted = True
            # else:
            #     permitted = c_a is c_b
            permitted = c_a is c_b
            comparison_permitted[key1] = permitted
            comparison_permitted[key2] = permitted


    comparison_permitted_by_label = {
        (a.__name__, b.__name__): v for ((a,b), v) in comparison_permitted.items()
    }

    print '|A|={0}, |B|={1}'.format(len([x for x in A.iter()]), len([x for x in B.iter()]))

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

    d = compare.simple_distance(A, B, simple_tree.Node.get_children, simple_tree.Node.get_label, comparison_filter=comparison_permitted_by_label)
    # d = compare.simple_distance(A, B, simple_tree.Node.get_children, simple_tree.Node.get_label, comparison_filter=None)

    t2 = datetime.datetime.now()
    print t2 - t1

    print 'Distance={0}'.format(d)


if __name__ == '__main__':
    test()

