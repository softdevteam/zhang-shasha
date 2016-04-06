#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author: Tim Henderson
#Email: tim.tadh@gmail.com
#For licensing see the LICENSE file in the top level directory.

import collections
import hashlib
import binascii


class Node(object):
    """
    A simple node object that can be used to construct trees to be used with
    :py:func:`zss.distance`.

    Example: ::

        Node("f")
            .addkid(Node("a")
                .addkid(Node("h"))
                .addkid(Node("c")
                    .addkid(Node("l"))))
            .addkid(Node("e"))
    """

    def __init__(self, label, value='', children=None, start=None, end=None, weight=1, original_node=None, compacted=False,
                 merge_id=None):
        self.label = label
        self.value = value
        self.parent = None
        self.children = children or list()
        for child in self.children:
            child.parent = self
        self.__sha = None
        self.__content_sha = None
        self.__fingerprint_index = None
        self.__content_fingerprint_index = None
        self.__depth = None
        self.__subtree_size = None
        self.original_node = original_node
        self.compacted = compacted
        self.start = start
        self.end = end
        self.weight = weight
        self.node_index = -1
        self.left_most_descendant_index = -1
        self.keyroot_node_index = -1
        self.keyroot_path_length = -1
        self.index_in_keyroot_list = -1
        self.dist_to_keyroot = -1
        self.merge_id = merge_id
        self.feature_vector = None
        self.left_sibling_feats = None
        self.right_sibling_feats = None
        self.left_tree_feats = None
        self.right_tree_feats = None
        self.matched = False
        self.best_match = None
        self.best_match_score = -1.0

    @staticmethod
    def get_children(node):
        """
        Default value of ``get_children`` argument of :py:func:`zss.distance`.

        :returns: ``self.children``.
        """
        return node.children

    @staticmethod
    def get_label(node):
        """
        Default value of ``get_label`` argument of :py:func:`zss.distance`.

        :returns: ``self.label``.
        """
        return node.label

    def addkid(self, node, before=False):
        """
        Add the given node as a child of this node.
        """
        if before:  self.children.insert(0, node)
        else:   self.children.append(node)
        node.parent = self
        return self

    def insert_child(self, pos, node):
        """
        Add a child at a specified position
        :param pos: position at which to insert child
        :param node: child node
        :return: self
        """
        self.children.insert(pos, node)
        node.parent = self
        return self

    def get(self, label):
        """:returns: Child with the given label."""
        if self.label == label: return self
        for c in self.children:
            if label in c: return c.get(label)

    def iter(self):
        """Iterate over this node and its children in a preorder traversal."""
        queue = collections.deque()
        queue.append(self)
        while len(queue) > 0:
            n = queue.popleft()
            for c in n.children: queue.append(c)
            yield n

    def iter_unmatched(self):
        """Iterate over this node and its children in a preorder traversal."""
        queue = collections.deque()
        queue.append(self)
        while len(queue) > 0:
            n = queue.popleft()
            for c in n.children:
                if not c.matched:
                    queue.append(c)
            yield n

    @property
    def sha(self):
        if self.__sha is None:
            cpt = '<{}>'.format(self.original_node.sha) if self.compacted  else ''
            hasher = hashlib.sha256()
            s = '{}{}({})'.format(self.label, cpt, ','.join(child.sha for child in self.children))
            hasher.update(s)
            self.__sha = binascii.hexlify(hasher.digest())
        return self.__sha

    @property
    def content_sha(self):
        if self.__content_sha is None:
            cpt = '<{}>'.format(self.original_node.content_sha) if self.compacted  else ''
            hasher = hashlib.sha256()
            s = '{}[{}]{}({})'.format(self.label, self.value, cpt, ','.join(child.sha for child in self.children))
            hasher.update(s)
            self.__content_sha = binascii.hexlify(hasher.digest())
        return self.__content_sha

    @property
    def depth(self):
        if self.__depth is None:
            self.__depth = 1 + (max([c.depth for c in self.children]) if len(self.children) > 0 else 0)
        return self.__depth

    @property
    def subtree_size(self):
        if self.__subtree_size is None:
            self.__subtree_size = 1 + sum([c.subtree_size for c in self.children])
        return self.__subtree_size

    @property
    def fingerprint_index(self):
        if self.__fingerprint_index is None:
            raise ValueError('Fingerprint not computed for node {0}'.format(id(self)))
        return self.__fingerprint_index

    @property
    def content_fingerprint_index(self):
        if self.__content_fingerprint_index is None:
            raise ValueError('Fingerprint not computed for node {0}'.format(id(self)))
        return self.__content_fingerprint_index

    def build_sha_table(self, sha_to_nodes):
        for child in self.children:
            child.build_sha_table(sha_to_nodes)
        nodes = sha_to_nodes.setdefault(self.sha, list())
        nodes.append(self)

    def update_fingerprint_index(self, sha_to_index, sha_indices_to_nodes, nodes_to_ignore=None):
        for child in self.children:
            child.update_fingerprint_index(sha_to_index, sha_indices_to_nodes, nodes_to_ignore=nodes_to_ignore)
        if nodes_to_ignore is None or self not in nodes_to_ignore:
            self.__fingerprint_index = sha_to_index.setdefault(self.sha, len(sha_to_index))
            nodes_for_index = sha_indices_to_nodes.setdefault(self.__fingerprint_index, list())
            nodes_for_index.append(self)

    def update_content_fingerprint_index(self, sha_to_index, sha_indices_to_nodes):
        for child in self.children:
            child.update_content_fingerprint_index(sha_to_index, sha_indices_to_nodes)
        self.__content_fingerprint_index = sha_to_index.setdefault(self.content_sha, len(sha_to_index))
        nodes_for_index = sha_indices_to_nodes.setdefault(self.__content_fingerprint_index, list())
        nodes_for_index.append(self)

    def all_nodes(self, node_list):
        for child in self.children:
            child.all_nodes(node_list)
        node_list.append(self)

    def fix_markers_top_down(self, start, end):
        start = self.start = self.start or start
        end = self.end = self.end or end

        # Get start markers for all children
        child_starts = [child.start for child in self.children]
        if len(child_starts) > 0:
            child_starts[0] = child_starts[0] or start

        # Fill in any blank entries backwards from the end
        valid_marker = end
        for i, s in reversed(list(enumerate(child_starts))):
            if s is None:
                child_starts[i] = valid_marker
            else:
                valid_marker = s

        # Start points are end points for previous elements
        child_ends = child_starts[1:] + [end]

        # Fix markers
        for child, start, end in zip(self.children, child_starts, child_ends):
            child.fix_markers_top_down(start, end)

    def fix_markers_bottom_up(self):
        if len(self.children) > 0:
            child_starts = [child.fix_markers_bottom_up() for child in self.children]
            child_starts = [s for s in child_starts if s is not None]
            if len(child_starts) > 0:
                s = min(child_starts)
                if self.start is None:
                    self.start = s
                else:
                    self.start = min(self.start, s)

        self.children.sort(key=lambda x: x.start)

        return self.start

    def check_markers(self):
        for a, b in zip(self.children[:-1], self.children[1:]):
            assert b.start >= a.end

        for c in self.children:
            assert c.end >= c.start

        for c in self.children:
            if c.start is not None and self.start is not None:
                assert c.start >= self.start

        for c in self.children:
            if c.end is not None and self.end is not None:
                assert c.end <= self.end

    def common_prefix_matches(self, matches, other_node, prefix_end_self, prefix_end_other):
        # If the node fingerprints match and their ranges are contained entirely within the common prefix,
        # add to the match list
        if self.fingerprint_index == other_node.fingerprint_index and \
                self.end <= prefix_end_self and other_node.end <= prefix_end_other:
            matches.append((self, other_node))
        # If the node range overlaps the common prefix
        if self.start <= prefix_end_self and other_node.start <= prefix_end_other:
            # Match child nodes
            for c_self, c_other in zip(self.children, other_node.children):
                if c_self.start > prefix_end_self or c_other.start > prefix_end_other:
                    break
                c_self.common_prefix_matches(matches, c_other, prefix_end_self, prefix_end_other)


    def common_suffix_matches(self, matches, other_node, suffix_start_self, suffix_start_other):
        # If the node fingerprints match and their ranges are contained entirely within the common suffix,
        # add to the match list
        if self.fingerprint_index == other_node.fingerprint_index and \
                self.start >= suffix_start_self and other_node.start >= suffix_start_other:
            matches.append((self, other_node))
        # If the node range overlaps the common suffix
        if self.end >= suffix_start_self and other_node.end >= suffix_start_other:
            for c_self, c_other in zip(reversed(self.children), reversed(other_node.children)):
                if c_self.end < suffix_start_self or c_other.end < suffix_start_other:
                    break
                c_self.common_suffix_matches(matches, c_other, suffix_start_self, suffix_start_other)


    def prune(self, start_marker, end_marker):
        children = []
        for child in self.children:
            if child.end >= start_marker and child.start <= end_marker:
                children.append(child.prune(start_marker, end_marker))
        return Node(label=self.label, children=children, start=self.start, end=self.end)

    def compact(self, nodes_to_compact):
        original_node = self.original_node if self.original_node is not None else self
        if self in nodes_to_compact:
            return Node(label=self.label, value=self.value, children=[],
                        start=self.start, end=self.end, weight=self.subtree_size, original_node=original_node, compacted=True,
                        merge_id=self.merge_id)
        else:
            return Node(label=self.label, value=self.value, children=[c.compact(nodes_to_compact) for c in self.children],
                        start=self.start, end=self.end, original_node=original_node, compacted=self.compacted,
                        merge_id=self.merge_id)

    def clone(self):
        original_node = self.original_node if self.original_node is not None else self
        return Node(label=self.label, value=self.value, children=[c.clone() for c in self.children],
                    start=self.start, end=self.end, original_node=original_node, compacted=self.compacted,
                    merge_id=self.merge_id)


    def update_node_list(self, node_list):
        for ch in self.children:
            ch.update_node_list(node_list)
        node_list.append(self)


    def _flatten_retained(self, flatten_pred_fn):
        ch = []
        for child in self.children:
            ch.extend(child._flatten(flatten_pred_fn))
        return Node(label=self.label, children=ch, start=self.start, end=self.end)

    def _flatten(self, flatten_pred_fn):
        if len(self.children) > 0 and flatten_pred_fn(self):
            return self.children
        else:
            return [self._flatten_retained(flatten_pred_fn)]

    def flatten(self, flatten_pred_fn):
        return self._flatten_retained(flatten_pred_fn)


    def content_compare(self, b):
        if not isinstance(b, Node):
            raise TypeError, "Must compare against type Node"
        if self.label != b.label or self.value != b.value or len(self.children) != len(b.children):
            return False
        for x, y in zip(self.children, b.children):
            if not x.content_compare(y):
                return False
        return True

    def shape_compare(self, b):
        if not isinstance(b, Node):
            raise TypeError, "Must compare against type Node"
        if self.label != b.label or len(self.children) != len(b.children):
            return False
        for x, y in zip(self.children, b.children):
            if not x.shape_compare(y):
                return False
        return True

    def __contains__(self, b):
        if isinstance(b, str) and self.label == b: return 1
        elif not isinstance(b, str) and self.label == b.label: return 1
        elif (isinstance(b, str) and self.label != b) or self.label != b.label:
            return sum(b in c for c in self.children)
        raise TypeError, "Object %s is not of type str or Node" % repr(b)

    def __eq__(self, b):
        if b is None: return False
        if not isinstance(b, Node):
            raise TypeError, "Must compare against type Node"
        return self.label == b.label

    def __ne__(self, b):
        return not self.__eq__(b)

    def __repr__(self):
        return super(Node, self).__repr__()[:-1] + " %s>" % self.label

    def __str__(self):
        s = "%d:%s" % (len(self.children), self.label)
        s = '\n'.join([s]+[str(c) for c in self.children])
        return s

    def pretty_print(self, level=0):
        if self.start is not None and self.end is not None:
            rng = ' ({0} - {1})'.format(self.start.pos, self.end.pos)
        elif self.start is not None and self.end is None:
            rng = ' ({0} -)'.format(self.start.pos)
        elif self.start is None and self.end is not None:
            rng = ' (- {0})'.format(self.end.pos)
        else:
            rng = ''
        valid = set()
        if self.end < self.start and self.start is not None and self.end is not None:
            valid.add(' ##**--SELF--**##')
        for c in self.children:
            if c.start < self.start and c.start is not None and self.start is not None:
                valid.add(' ##**--CH_START--**##')
            if c.end > self.end and c.end is not None and self.end is not None:
                valid.add(' ##**--CH_END--**##')
        for c0, c1 in zip(self.children[:-1], self.children[1:]):
            if c0.end > c1.start and c0.end is not None and c1.start is not None:
                valid.add(' ##**--CH_ORD--**##')
        valid = ''.join(sorted(valid))
        if len(self.children) > 0:
            ch = '\n'.join([c.pretty_print(level+1) for c in self.children])
            return '{0}{1}{2}{3}:\n{4}'.format('  ' * level, self.label, valid, rng, ch)
        else:
            return '{0}{1}{2}{3}'.format('  ' * level, self.label, valid, rng)
