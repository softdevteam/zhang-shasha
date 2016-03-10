import _ast, type_pruning

def _retain(cls):
    if issubclass(cls, (_ast.stmt, _ast.mod)):
        return True
    elif cls in {_ast.List, _ast.Dict, _ast.Set, _ast.Tuple,
                 _ast.DictComp, _ast.ListComp, _ast.SetComp, _ast.GeneratorExp, _ast.ExceptHandler, _ast.Call}:
        return True
    else:
        return False


def flatten_types():
    node_classes = [getattr(_ast, name) for name in dir(_ast)]
    node_classes = [cls for cls in node_classes
                    if isinstance(cls, type) and issubclass(cls, _ast.AST) and type_pruning.ast_base(cls) is not _ast.AST]
    node_classes.sort(key=lambda x: x.__name__)

    retained = [cls for cls in node_classes if _retain(cls)]

    flattened = [cls for cls in node_classes if cls not in retained]

    return flattened, retained


if __name__ == '__main__':
    flattened, retained = flatten_types()

    print 'RETAINED:'
    for cls in retained:
        print cls
    print 'FLATTENED:'
    for cls in flattened:
        print cls

