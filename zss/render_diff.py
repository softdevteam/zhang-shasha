import edit_script, cgi
import flask



_op_to_colour = {
    edit_script.InsertEditOp: "diff_ins",
    edit_script.RemoveEditOp: "diff_del",
    edit_script.UpdateEditOp: "diff_upd",
    None: "diff_match",
}


def _add_to_source_result(result, src, start_marker, end_marker, op, depth):
    text = src[start_marker:end_marker]
    result.append(cgi.escape(text))


def _render_source_node(result, src, tree_node, node_ops, depth):
    op = node_ops.get(tree_node)
    result.append('<span class="{}">'.format(_op_to_colour[op]))
    left_marker = tree_node.start
    for c in tree_node.children:
        right_marker = c.start
        if right_marker > left_marker:
            _add_to_source_result(result, src, left_marker, right_marker, op, depth)

        _render_source_node(result, src, c, node_ops, depth + 1)

        left_marker = c.end

    right_marker = tree_node.end
    if right_marker > left_marker:
        _add_to_source_result(result, src, left_marker, right_marker, op, depth)
    result.append('</span>'.format(_op_to_colour[op]))


def _render_source_diffs(src, tree, merge_id_to_node, diffs):
    result = []
    node_ops = {}
    for diff in diffs:
        diff.build_node_to_op_table(node_ops, merge_id_to_node)
    _render_source_node(result, src, tree, node_ops, 0)
    result = ''.join(result)
    return '<pre>' + result + '</pre>'


def _render_tree_node(result, tree_node, node_ops):
    op = node_ops.get(tree_node)
    result.append('<li class="node_header {}">{}({})</li>'.format(_op_to_colour[op], tree_node.label, tree_node.value))
    result.append('<ul class="node_children">')
    for child in tree_node.children:
        _render_tree_node(result, child, node_ops)
    result.append('</ul>')

def _render_tree_diffs(tree, merge_id_to_node, diffs):
    result = []
    node_ops = {}
    for diff in diffs:
        diff.build_node_to_op_table(node_ops, merge_id_to_node)
    result.append('<ul class="tree_root">')
    _render_tree_node(result, tree, node_ops)
    result.append('</ul>')
    result = ''.join(result)
    return result


def render_diffs(A_src, A, B_src, B, diffs, header):
    merge_id_to_node_a = {}
    merge_id_to_node_b = {}
    for a in A.iter():
        merge_id_to_node_a[a.merge_id] = a
    for b in B.iter():
        merge_id_to_node_b[b.merge_id] = b

    diffs_left = _render_source_diffs(A_src, A, merge_id_to_node_a, diffs)
    diffs_right = _render_source_diffs(B_src, B, merge_id_to_node_b, diffs)

    tree_left = _render_tree_diffs(A, merge_id_to_node_a, diffs)
    tree_right = _render_tree_diffs(B, merge_id_to_node_b, diffs)

    return flask.render_template("diffs.jinja2", header=header, diffs_left=diffs_left, diffs_right=diffs_right,
                                 tree_left=tree_left, tree_right=tree_right)


