import edit_script, cgi


_html_doc_template = """
<html>
	<head>
		<title>Diff</title>
		<style>
			.overall {
				background: #f8f8f8;
			}

			.diff {
				border-radius: 5px;
				border: 1px solid #e0e0e0;
				background: #ffffff;
			}

			.left_container {
				width: 49%;
				padding: 5px;
			}

			.right_container {
				width: 49%;
				left: 50%;
				position: absolute;
				padding: 5px;
			}

			.diff_ins {
				background: #c0ffc0;
			}

			.diff_del {
				background: #ffc0c0;
			}

			.diff_upd {
				background: #ffff80;
			}

		</style>
	</head>

	<body>
		<p>The diffs</p>

		<div class="overall">
			<div class="right_container"><div class="diff">##RIGHT##</div></div>
			<div class="left_container"><div class="diff">##LEFT##</div></div>
		</div>
	</body>
</html>
"""


_op_to_colour = {
    edit_script.InsertEditOp: "#c0ffc0",
    edit_script.RemoveEditOp: "#ffc0c0",
    edit_script.UpdateEditOp: "#ffffc0",
    None: "#ffffff",
}


def _add_to_result(result, src, start_marker, end_marker, op, depth):
    text = src[start_marker:end_marker]
    html = '<span style="background:{}">{}</span>'.format(_op_to_colour[op], cgi.escape(text))
    result.append(html)


def _render_node(result, src, tree_node, node_ops, depth):
    op = node_ops.get(tree_node)
    left_marker = tree_node.start
    for c in tree_node.children:
        right_marker = c.start
        if right_marker > left_marker:
            _add_to_result(result, src, left_marker, right_marker, op, depth)

        _render_node(result, src, c, node_ops, depth+1)

        left_marker = c.end

    right_marker = tree_node.end
    if right_marker > left_marker:
        _add_to_result(result, src, left_marker, right_marker, op, depth)


def _render_source_diffs(src, tree, merge_id_to_node, diffs):
    result = []
    node_ops = {}
    for diff in diffs:
        diff.build_node_to_op_table(node_ops, merge_id_to_node)
    _render_node(result, src, tree, node_ops, 0)
    result = ''.join(result)
    return '<pre>' + result + '</pre>'


def render_diffs(A_src, A, B_src, B, diffs):
    merge_id_to_node_a = {}
    merge_id_to_node_b = {}
    for a in A.iter():
        merge_id_to_node_a[a.merge_id] = a
    for b in B.iter():
        merge_id_to_node_b[b.merge_id] = b

    da = _render_source_diffs(A_src, A, merge_id_to_node_a, diffs)
    db = _render_source_diffs(B_src, B, merge_id_to_node_b, diffs)

    return _html_doc_template.replace('##LEFT##', da).replace('##RIGHT##', db)


