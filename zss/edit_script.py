import simple_tree


class EditOp (object):
    def apply(self, node_by_index):
        raise NotImplementedError('abstact for type {0}'.format(type(self)))


class RemoveEditOp (EditOp):
    def __init__(self, node_id):
        self.node_id = node_id

    def apply(self, merge_id_to_node):
        node = merge_id_to_node[self.node_id]
        parent = node.parent
        parent.children.remove(node)


class InsertEditOp (EditOp):
    def __init__(self, parent_node_id, node_id, pos, label, value):
        self.parent_node_id = parent_node_id
        self.node_id = node_id
        self.pos = pos
        self.label = label
        self.value = value

    def apply(self, merge_id_to_node):
        parent = merge_id_to_node[self.parent_node_id]
        node = simple_tree.Node(label=self.label, value=self.value, children=[], merge_id=self.node_id)
        parent.insert_child(self.pos, node)
        merge_id_to_node[self.node_id] = node


class UpdateEditOp (EditOp):
    def __init__(self, node_id, value):
        self.node_id = node_id
        self.value = value

    def apply(self, merge_id_to_node):
        node = merge_id_to_node[self.node_id]
        node.value = self.value


def edit_script(A, B, node_match_pairs):
    diffs = []

    matched_a = set()
    matched_b = set()
    merge_id = 0

    # Walk the matched node pairs, assign merge IDs and add update operations as necessary
    for k, (a_i, b_j) in enumerate(node_match_pairs):
        a_i.merge_id = b_j.merge_id = merge_id
        matched_a.add(a_i)
        matched_b.add(b_j)

        if a_i.value != b_j.value:
            diffs.append(UpdateEditOp(k, b_j.value))
        merge_id += 1

    # Find unmatched nodes in A; deleted nodes, assign merge IDs and create diffs
    remove_diffs = []
    for a in A.iter():
        if a not in matched_a:
            a.merge_id = merge_id
            remove_diffs.append(RemoveEditOp(merge_id))
            merge_id += 1
    remove_diffs.reverse()

    # Find unmatched nodes in B; inserted nodes, assign merge IDs and create diffs
    for b in B.iter():
        if b not in matched_b:
            b.merge_id = merge_id
            diffs.append(InsertEditOp(b.parent.merge_id, merge_id, b.parent.children.index(b), b.label, b.value))
            merge_id += 1

    return remove_diffs + diffs

