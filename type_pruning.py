import collections
import _ast

import python_ast_structure


def ast_base(cls):
    return cls.__bases__[0]


def ast_to_type_containment(ast, type_containment=None):
    """
    Generate a dictionary that contains direct parent to child type relationships, e.g. if in the provided AST,
    nodes of type `_ast.Return` have children of type `_ast.Name` and `_ast.Call`, then the returned
    dictionary will contain an entry `_ast.Return: {_ast.Name, _ast.Call}`

    :param ast: an abstract syntax tree from which type relationships should be extracted
    :param type_containment: an existing type containment map if you want to extend an existing one rather
    than build a new one
    :return: a dictionary mapping ast type `parent_type` to a set of ast types `child_types` that are the types
    of nodes that are child of node of type `parent_type`; `{parent_type: {child_type1, child_type2, ...}}`
    """
    if type_containment is None:
        type_containment = {}
    node_stack = [ast]
    while len(node_stack) > 0:
        node = node_stack.pop()
        children = []
        for field_name in node._fields:
            field_val = getattr(node, field_name)
            if isinstance(field_val, (list, tuple)):
                children.extend([x for x in field_val if isinstance(x, _ast.AST)])
            elif isinstance(field_val, _ast.AST):
                children.append(field_val)
        ts = type_containment.setdefault(type(node), set())
        for child in children:
            ts.add(type(child))
            node_stack.append(child)
    return type_containment



def compute_node_type_compatibility_map_equality(node_classes):
    """
    Create a node type compatibility map for which a node type is only compatible with itself.

    :param node_classes: the list of node classes
    :return: a dictionary mapping {(type1, type2): bool}
    """
    comparison_permitted = {}

    for i in xrange(len(node_classes)):
        c_a = node_classes[i]
        for j in xrange(i+1):
            c_b = node_classes[j]
            key1 = c_a, c_b
            key2 = c_b, c_a
            permitted = c_a is c_b
            comparison_permitted[key1] = permitted
            comparison_permitted[key2] = permitted

    return comparison_permitted


def compute_node_type_compatibility_map_by_base(node_classes):
    """
    Create a node type compatibility map for which a node type is only compatible types that inherit from the
    same base class, with the exception of function definitions (_ast.FunctionDef) and class definitions (_ast.ClassDef).

    :param node_classes: the list of node classes
    :return: a dictionary mapping {(type1, type2): bool}
    """
    special_classes = {_ast.FunctionDef, _ast.ClassDef}

    comparison_permitted = {}

    for i in xrange(len(node_classes)):
        c_a = node_classes[i]
        for j in xrange(i+1):
            c_b = node_classes[j]
            key1 = c_a, c_b
            key2 = c_b, c_a
            if c_a in special_classes or c_b in special_classes:
                permitted = c_a is c_b
            elif ast_base(c_a) is ast_base(c_b):
                permitted = True
            else:
                permitted = c_a is c_b
            comparison_permitted[key1] = permitted
            comparison_permitted[key2] = permitted

    return comparison_permitted


def compute_node_type_compatibility_map_by_containment(node_classes, type_containment_map):
    """
    Create a node type compatibility map based on previously gathered type containment relationships.
    The type containment map can be created from an ast using the `ast_to_type_containment` function.

    :param node_classes: the list of node classes
    :param type_containment_map: a dictionary of the form `{parent_type: {child_type1, child_type2, ...}}`
    :return: a dictionary mapping {(type1, type2): bool}
    """
    type_containment = {}

    for container, contained in type_containment_map.items():
        all_contained = set()
        stack = collections.deque()
        stack.extend(list(contained))
        while len(stack) > 0:
            c = stack.popleft()
            if c not in all_contained:
                all_contained.add(c)
                stack.append(c)
        type_containment[container] = all_contained

    comparison_permitted = {}

    for i in xrange(len(node_classes)):
        c_a = node_classes[i]
        for j in xrange(i+1):
            c_b = node_classes[j]
            key1 = c_a, c_b
            key2 = c_b, c_a
            if c_a is c_b:
                permitted = True
            elif c_a in type_containment.get(c_b, set()):
                permitted = True
            elif c_b in type_containment.get(c_a, set()):
                permitted = True
            else:
                permitted = False
            comparison_permitted[key1] = permitted
            comparison_permitted[key2] = permitted

    return comparison_permitted


_node_type_contain = {node_type: python_ast_structure.spec_to_type_set(spec)
                            for node_type, spec in python_ast_structure.ast_type_to_spec.items()}


def compute_node_type_compatibility_by_grammar(node_classes):
    """
    Create a node type compatibility map according to the AST node containments permitted by the Python grammar

    :param node_classes: the list of node classes
    :param type_containment_map: a dictionary of the form `{parent_type: {child_type1, child_type2, ...}}`
    :return: a dictionary mapping {(type1, type2): bool}
    """
    # Generate a dictionary that maps AST base types to the set of their subclasses
    subs_by_base = {}
    for nc in node_classes:
        if nc is not _ast.AST:
            b = ast_base(nc)
            if b is not _ast.AST:
                subs = subs_by_base.setdefault(b, set())
                subs.add(nc)

    # Define a function that will recursive replace a class with derived classes until this cannot be done further
    def base_to_subs(c):
        return list(subs_by_base.get(c, {c}))


    # The type_containment dictionary is the transitive closure of the graph defined by
    # `_node_type_contain` that maps node types to lists of base types and the
    # `base_to_subs` function that maps bases back to node types
    type_containment = {}

    for container in _node_type_contain.keys():
        all_contained = set()
        visited = set()
        stack = collections.deque()
        stack.append(container)
        # Iterative algorithm to repeatedly follow the graph edges until the transitive closure
        # has been traversed
        # print '*** {0} -> {1}'.format(container, stack)
        while len(stack) > 0:
            c = stack.popleft()

            # `c` is a concrete node type
            all_contained.add(c)

            # Get the base types that can be contained by `c`
            bases_within_c = _node_type_contain.get(c, set())

            for base_within_c in bases_within_c:
                # For each base type, get the concrete types that derive from it
                for concrete_in_c in base_to_subs(base_within_c):
                    # Visit all concrete types
                    if concrete_in_c not in visited:
                        all_contained.add(concrete_in_c)
                        visited.add(concrete_in_c)
                        stack.append(concrete_in_c)

        type_containment[container] = all_contained

    comparison_permitted = {}

    for i in xrange(len(node_classes)):
        c_a = node_classes[i]
        for j in xrange(i+1):
            c_b = node_classes[j]
            key1 = c_a, c_b
            key2 = c_b, c_a
            if c_a is c_b:
                permitted = True
            else:
                in_a = type_containment.get(c_a, set())
                in_b = type_containment.get(c_b, set())
                permitted = len(in_a.intersection(in_b)) > 0
            comparison_permitted[key1] = permitted
            comparison_permitted[key2] = permitted

    return comparison_permitted


def type_compatbility_map_to_matrix(node_classes, type_compatibility_map):
    lines = []
    for a in node_classes:
        line = ''.join([('*' if type_compatibility_map[(a,b)] else '.') for b in node_classes])
        lines.append(line)
    return '\n'.join(lines)
