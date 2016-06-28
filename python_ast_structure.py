import _ast, types

def _maybe(t):
    return (t, types.NoneType)

_number = (int, long, float, complex)

ast_type_to_spec = {
    # mod
    _ast.Module: dict(body=[_ast.stmt]),
    _ast.Interactive: dict(body=[_ast.stmt]),
    _ast.Expression: dict(body=_ast.expr),
    _ast.Suite: dict(body=[_ast.stmt]),

    # stmt
    _ast.FunctionDef: dict(name=str, args=_ast.arguments, body=[_ast.stmt], decorator_list=[_ast.expr]),
    _ast.ClassDef: dict(name=str, bases=[_ast.expr], body=[_ast.stmt], decorator_list=[_ast.expr]),
    _ast.Return: dict(value=_maybe(_ast.expr)),
    _ast.Delete: dict(targets=[_ast.expr]),
    _ast.Assign: dict(targets=[_ast.expr], value=_ast.expr),
    _ast.AugAssign: dict(target=_ast.expr, op=_ast.operator, value=_ast.expr),
    _ast.Print: dict(dest=_maybe(_ast.expr), values=[_ast.expr], nl=bool),
    _ast.For: dict(target=_ast.expr, iter=_ast.expr, body=[_ast.stmt], orelse=[_ast.stmt]),
    _ast.While: dict(test=_ast.expr, body=[_ast.stmt], orelse=[_ast.stmt]),
    _ast.If: dict(test=_ast.expr, body=[_ast.stmt], orelse=[_ast.stmt]),
    _ast.With: dict(context_expr=_ast.expr, optional_vars=_maybe(_ast.expr), body=[_ast.stmt]),
    _ast.Raise: dict(type=_maybe(_ast.expr), inst=_maybe(_ast.expr), tback=_maybe(_ast.expr)),
    _ast.TryExcept: dict(body=[_ast.stmt], handlers=[_ast.excepthandler], orelse=[_ast.stmt]),
    _ast.TryFinally: dict(body=[_ast.stmt], finalbody=[_ast.stmt]),
    _ast.Assert: dict(test=_ast.expr, msg=_maybe(_ast.expr)),
    _ast.Import: dict(names=[_ast.alias]),
    _ast.ImportFrom: dict(module=_maybe(str), names=[_ast.alias], level=_maybe(int)),
    _ast.Exec: dict(body=_ast.expr, globals=_maybe(_ast.expr), locals=_maybe(_ast.expr)),
    _ast.Global: dict(names=[str]),
    _ast.Expr: dict(value=_ast.expr),
    _ast.Pass: dict(),
    _ast.Break: dict(),
    _ast.Continue: dict(),

    # expr
    _ast.BoolOp: dict(op=_ast.boolop, values=[_ast.expr]),
    _ast.BinOp: dict(left=_ast.expr, op=_ast.operator, right=_ast.expr),
    _ast.UnaryOp: dict(op=_ast.unaryop, operand=_ast.expr),
    _ast.Lambda: dict(args=_ast.arguments, body=_ast.expr),
    _ast.IfExp: dict(test=_ast.expr, body=_ast.expr, orelse=_ast.expr),
    _ast.Dict: dict(keys=[_ast.expr], values=[_ast.expr]),
    _ast.Set: dict(elts=[_ast.expr]),
    _ast.ListComp: dict(elt=_ast.expr, generators=[_ast.comprehension]),
    _ast.SetComp: dict(elt=_ast.expr, generators=[_ast.comprehension]),
    _ast.DictComp: dict(key=_ast.expr, value=_ast.expr, generators=[_ast.comprehension]),
    _ast.GeneratorExp: dict(elt=_ast.expr, generators=[_ast.comprehension]),
    _ast.Yield: dict(value=_maybe(_ast.expr)),
    _ast.Compare: dict(left=_ast.expr, ops=[_ast.cmpop], comparators=[_ast.expr]),
    _ast.Call: dict(func=_ast.expr, args=[_ast.expr], keywords=[_ast.keyword],
                    starargs=_maybe(_ast.expr), kwargs=_maybe(_ast.expr)),
    _ast.Repr: dict(value=_ast.expr),
    _ast.Num: dict(n=_number),
    _ast.Str: dict(s=str),
    _ast.Attribute: dict(value=_ast.expr, attr=str, ctx=_ast.expr_context),
    _ast.Subscript: dict(value=_ast.expr, slice=_ast.slice, ctx=_ast.expr_context),
    _ast.Name: dict(id=str, ctx=_ast.expr_context),
    _ast.List: dict(elts=[_ast.expr], ctx=_ast.expr_context),
    _ast.Tuple: dict(elts=[_ast.expr], ctx=_ast.expr_context),

    # expr_context
    _ast.Load: dict(),
    _ast.Store: dict(),
    _ast.Del: dict(),
    _ast.AugLoad: dict(),
    _ast.AugStore: dict(),
    _ast.Param: dict(),

    # slice
    _ast.Ellipsis: dict(),
    _ast.Slice: dict(lower=_maybe(_ast.expr), upper=_maybe(_ast.expr), step=_maybe(_ast.expr)),
    _ast.ExtSlice: dict(dims=[_ast.slice]),
    _ast.Index: dict(value=_ast.expr),

    # boolop
    _ast.And: dict(),
    _ast.Or: dict(),

    # operator
    _ast.Add: dict(),
    _ast.Sub: dict(),
    _ast.Mult: dict(),
    _ast.Div: dict(),
    _ast.Mod: dict(),
    _ast.Pow: dict(),
    _ast.LShift: dict(),
    _ast.RShift: dict(),
    _ast.BitOr: dict(),
    _ast.BitXor: dict(),
    _ast.BitAnd: dict(),
    _ast.FloorDiv: dict(),

    # unaryop
    _ast.Invert: dict(),
    _ast.Not: dict(),
    _ast.UAdd: dict(),
    _ast.USub: dict(),

    # cmpop
    _ast.Eq: dict(),
    _ast.NotEq: dict(),
    _ast.Lt: dict(),
    _ast.LtE: dict(),
    _ast.Gt: dict(),
    _ast.GtE: dict(),
    _ast.Is: dict(),
    _ast.IsNot: dict(),
    _ast.In: dict(),
    _ast.NotIn: dict(),

    # comprehension
    _ast.comprehension: dict(target=_ast.expr, iter=_ast.expr, ifs=[_ast.expr]),

    # excepthanlder
    _ast.ExceptHandler: dict(type=_maybe(_ast.expr), name=_maybe(_ast.expr), body=[_ast.stmt]),

    # arguments
    _ast.arguments: dict(args=[_ast.expr], vararg=_maybe(str), kwarg=_maybe(str), defaults=[_ast.expr]),

    # keyword
    _ast.keyword: dict(arg=str, value=_ast.expr),

    # alias
    _ast.alias: dict(name=str, asname=_maybe(str)),
}

def spec_to_type_set(spec):
    if isinstance(spec, dict):
        types = set()
        for t in spec.values():
            if isinstance(t, list):
                if len(t) != 1:
                    raise TypeError('Lists in type specs should indicate a list '
                                    'of a given type and should have only 1 element, not {0}'.format(len(t)))
                types.update(spec_to_type_set(t[0]))
            elif isinstance(t, tuple):
                for x in t:
                    types.update(spec_to_type_set(x))
            elif isinstance(t, type) or issubclass(t, _ast.AST):
                types.add(t)
            else:
                raise TypeError('Values in a type spec dictionary should be either '
                                'types, lists or tuples, not {0}'.format(t))
        return types
    elif isinstance(spec, type) or issubclass(spec, _ast.AST):
        return {spec}
    else:
        raise TypeError('A type spec should be either a dictionary or a type, not {0}'.format(spec))


def _make_ast_class_name_to_id_map():
    classes = ast_type_to_spec.keys()
    cls_names = [cls.__name__ for cls in classes]
    cls_names.sort()
    return {name: i for i, name in enumerate(cls_names)}

ast_class_name_to_id_map = _make_ast_class_name_to_id_map()



import unittest

class Test_ast_structure (unittest.TestCase):
    def test_type_to_spec_fields(self):
        # Verify the fields
        for ast_type, spec in ast_type_to_spec.items():
            fields = list(ast_type._fields)
            for name, val_type in spec.items():
                self.assertIn(name, fields)