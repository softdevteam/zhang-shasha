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

    def __init__(self, label, children=None, start=None, end=None):
        self.label = label
        self.children = children or list()
        self.__sha = None
        self.__fingerprint_index = None
        self.__depth = None
        self.__subtree_size = None
        self.start = start
        self.end = end
        self.nidx = -1
        self.kridx = -1

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

    @property
    def sha(self):
        if self.__sha is None:
            hasher = hashlib.sha256()
            s = '{0}({1})'.format(self.label, ','.join(child.sha for child in self.children))
            hasher.update(s)
            self.__sha = binascii.hexlify(hasher.digest())
        return self.__sha

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

    def build_sha_table(self, sha_to_nodes):
        for child in self.children:
            child.build_sha_table(sha_to_nodes)
        nodes = sha_to_nodes.setdefault(self.sha, list())
        nodes.append(self)

    def update_fingerprint_index(self, sha_to_index):
        for child in self.children:
            child.update_fingerprint_index(sha_to_index)
        self.__fingerprint_index = sha_to_index.setdefault(self.sha, len(sha_to_index))

    def all_nodes(self, node_list):
        for child in self.children:
            child.all_nodes(node_list)
        node_list.append(self)

    def fix_markers_top_down(self, start, end):
        start = self.start = self.start or start
        end = self.end = self.end or end

        # Get start markers for all children
        child_starts = [child.start for child in self.children]

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

        for a, b in zip(self.children[:-1], self.children[1:]):
            assert b.start >= a.end

    def fix_markers_bottom_up(self):
        if len(self.children) > 0:
            child_starts = [child.fix_markers_bottom_up() for child in self.children]
            if self.start is None:
                child_starts = [s for s in child_starts if s is not None]
                if len(child_starts) > 0:
                    s = min(child_starts)
                    self.start = s

        return self.start


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
        if len(self.children) > 0:
            ch = '\n'.join([c.pretty_print(level+1) for c in self.children])
            return '{0}{1}{2}:\n{3}'.format('  ' * level, self.label, rng, ch)
        else:
            return '{0}{1}{2}'.format('  ' * level, self.label, rng)
