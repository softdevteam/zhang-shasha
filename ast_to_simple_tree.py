import ast

from zss import simple_tree
import source_text

def tuple_to_tree(x):
    label = x[0]
    return simple_tree.Node(str(label), [tuple_to_tree(y) for y in x[1:] if isinstance(y, tuple)])


class NodeClass (object):
    def __init__(self, cls_name):
        self.cls_name = cls_name

    def node(self, value, children, start=None, end=None):
        return simple_tree.Node(self.cls_name, value, children, start=start, end=end)


class ASTConverter (object):
    """
    Python AST to simple_tree.Node converter.
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

    def _handle_ast_value(self, src_text, children, values, x):
        if isinstance(x, ast.AST):
            children.append(self._ast_to_tree(src_text, x))
        elif isinstance(x, (list, tuple)):
            for v in x:
                self._handle_ast_value(src_text, children, values, v)
        else:
            values.append(str(x))

    def _ast_node_marker(self, src_text, ast_node):
        if hasattr(ast_node, 'lineno'):
            return src_text.marker_at_loc(ast_node.lineno-1, ast_node.col_offset)
        else:
            return None

    def _ast_to_tree(self, src_text, ast_node):
        children = []
        values = []
        for field_name in ast_node._fields:
            field_val = getattr(ast_node, field_name)
            self._handle_ast_value(src_text, children, values, field_val)
        node_class = self._get_node_class(ast_node)
        value = '_'.join([str(x) for x in values])
        children.sort(key=lambda x: x.start)
        return node_class.node(value, children, start=self._ast_node_marker(src_text, ast_node))

    def parse(self, code):
        if not isinstance(code, source_text.SourceText):
            code = source_text.SourceText(code)
        a = ast.parse(code.text)
        t = self._ast_to_tree(code, a)
        t.fix_markers_bottom_up()
        t.fix_markers_top_down(code.marker_at_start(), code.marker_at_end())
        return t

    def node_classes(self):
        return self._node_classes.keys()
