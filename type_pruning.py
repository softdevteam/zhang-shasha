import collections
import _ast


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


_node_type_contain = {
    # mod
    _ast.Module: {_ast.stmt},
    _ast.Interactive: {_ast.stmt},
    _ast.Expression: {_ast.expr},
    _ast.Suite: {_ast.stmt},

    # stmt
    _ast.FunctionDef: {_ast.expr, _ast.stmt},
    _ast.ClassDef: {_ast.expr, _ast.stmt},
    _ast.Return: {_ast.expr},
    _ast.Delete: {_ast.expr},
    _ast.Assign: {_ast.expr},
    _ast.AugAssign: {_ast.expr, _ast.operator},
    _ast.Print: {_ast.expr},
    _ast.For: {_ast.expr, _ast.stmt},
    _ast.While: {_ast.expr, _ast.stmt},
    _ast.If: {_ast.expr, _ast.stmt},
    _ast.With: {_ast.expr, _ast.stmt},
    _ast.Raise: {_ast.expr},
    _ast.TryExcept: {_ast.stmt, _ast.excepthandler},
    _ast.TryFinally: {_ast.stmt},
    _ast.Assert: {_ast.expr},
    _ast.Import: {},
    _ast.ImportFrom: {},
    _ast.Exec: {_ast.expr},
    _ast.Global: set(),
    _ast.Expr: {_ast.expr},
    _ast.Pass: {_ast.expr},
    _ast.Break: {_ast.expr},
    _ast.Continue: {_ast.expr},

    # expr
    _ast.BoolOp: {_ast.expr},
    _ast.BinOp: {_ast.expr, _ast.operator},
    _ast.UnaryOp: {_ast.expr, _ast.unaryop},
    _ast.Lambda: {_ast.expr},
    _ast.IfExp: {_ast.expr},
    _ast.Dict: {_ast.expr},
    _ast.Set: {_ast.expr},
    _ast.ListComp: {_ast.expr},
    _ast.SetComp: {_ast.expr},
    _ast.DictComp: {_ast.expr},
    _ast.GeneratorExp: {_ast.expr},
    _ast.Yield: {_ast.expr},
    _ast.Compare: {_ast.expr, _ast.cmpop},
    _ast.Call: {_ast.expr},
    _ast.Repr: {_ast.expr},
    _ast.Num: {},
    _ast.Str: {},
    _ast.Attribute: {_ast.expr, _ast.expr_context},
    _ast.Subscript: {_ast.expr, _ast.expr_context, _ast.slice},
    _ast.Name: {_ast.expr_context},
    _ast.List: {_ast.expr, _ast.expr_context},
    _ast.Tuple: {_ast.expr, _ast.expr_context},

    # expr_context
    _ast.Load: {},
    _ast.Store: {},
    _ast.Del: {},
    _ast.AugLoad: {},
    _ast.AugStore: {},
    _ast.Param: {},

    # slice
    _ast.Ellipsis: {},
    _ast.Slice: {_ast.expr},
    _ast.ExtSlice: {_ast.slice},
    _ast.Index: {_ast.Expression},

    # boolop
    _ast.And: {},
    _ast.Or: {},

    # operator
    _ast.Add: {},
    _ast.Sub: {},
    _ast.Mult: {},
    _ast.Div: {},
    _ast.Mod: {},
    _ast.Pow: {},
    _ast.LShift: {},
    _ast.RShift: {},
    _ast.BitOr: {},
    _ast.BitXor: {},
    _ast.BitAnd: {},
    _ast.FloorDiv: {},

    # unaryop
    _ast.Invert: {},
    _ast.Not: {},
    _ast.UAdd: {},
    _ast.USub: {},

    # cmpop
    _ast.Eq: {},
    _ast.NotEq: {},
    _ast.Lt: {},
    _ast.LtE: {},
    _ast.Gt: {},
    _ast.GtE: {},
    _ast.Is: {},
    _ast.IsNot: {},
    _ast.In: {},
    _ast.NotIn: {},

}

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
