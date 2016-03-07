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

    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()
        self.__sha = None
        self.fingerprint_index = None
        self.__depth = None
        self.__subtree_size = None

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

    def build_sha_table(self, sha_to_nodes):
        for child in self.children:
            child.build_sha_table(sha_to_nodes)
        nodes = sha_to_nodes.setdefault(self.sha, list())
        nodes.append(self)

    def update_fingerprint_index(self, sha_to_index):
        for child in self.children:
            child.update_fingerprint_index(sha_to_index)
        self.fingerprint_index = sha_to_index.setdefault(self.sha, len(sha_to_index))

    def all_nodes(self, node_list):
        for child in self.children:
            child.all_nodes(node_list)
        node_list.append(self)

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
