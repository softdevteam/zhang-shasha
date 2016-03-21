#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Authors: Tim Henderson and Steve Johnson
#Email: tim.tadh@gmail.com, steve@steveasleep.com
#For licensing see the LICENSE file in the top level directory.

import collections, itertools

try:
    import numpy as np
    zeros = np.zeros
except ImportError:
    def py_zeros(dim, pytype):
        assert len(dim) == 2
        return [[pytype() for y in xrange(dim[1])]
                for x in xrange(dim[0])]
    zeros = py_zeros

try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1

from zss.simple_tree import Node


class AnnotatedTree(object):

    def __init__(self, root, get_children):
        self.get_children = get_children

        self.root = root
        self.nodes = list()  # a pre-order enumeration of the nodes in the tree
        self.ids = list()    # a matching list of ids
        self.lmds = list()   # left most descendents
        self.keyroots = None
            # k and k' are nodes specified in the pre-order enumeration.
            # keyroots = {k | there exists no k'>k such that lmd(k) == lmd(k')}
            # see paper for more on keyroots

        stack = list()
        pstack = list()
        stack.append((root, collections.deque()))
        j = 0
        while len(stack) > 0:
            n, anc = stack.pop()
            nid = j
            # children of `n` get pushed into `stack` left-to-right,
            # so they are popped off right-to-left
            for c in self.get_children(n):
                a = collections.deque(anc)
                a.appendleft(nid)
                stack.append((c, a))
            pstack.append(((n, nid), anc))
            j += 1
        # At this point, `pstack` contains a right-to-left pre-order traversal of the tree
        # So: the first item is the root, the last is the left-most leaf
        # where each item is ((node, node_id), ancestor_node_ids)
        # where node_ids are the indices of the node to which they refer in `pstack`

        # The loop below pops nodes off in reverse order, so we will end up building a list of nodes
        # that is a left-to-right post-order traversal:
        # The first node is the left-most-leaf, the last is the root

        # `self.nodes` is a list of nodes in order of left-to-right post-order traversal
        # `self.ids` is a list of node ids that provides the node ID of each node in `self.nodes`

        # `self.lmds` is a list of indices that index into `self.nodes`
        # It provides the left-most-descendant of each node
        # self.lmds[i] == self.nodes.index(left_most_descendant(self.nodes[i]))
        #
        # `self.keyroots` is also a list of indices that index into `self.nodes`
        # It lists the indices of keyroot nodes
        # For each keyroot node `k` there exists no keyroot node `j` that has the same left-most-descendant
        # as `k` where `j` is a descendant of `k`, in other words, each `k` in `self.keyroots` has
        # a different left-most-descendant, and `k` is the furthest ancestor from
        # `left_most_descendant(k)`
        # `self.keyroots` is in order of left-to-right post-order traversal

        lmds = dict()
        keyroots = dict()
        i = 0
        while len(pstack) > 0:
            (n, nid), anc = pstack.pop()
            #print list(anc)
            n.nidx = len(self.nodes)
            self.nodes.append(n)
            self.ids.append(nid)
            #print n.label, [a.label for a in anc]
            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in lmds:
                        lmds[a] = i
                    else:
                        break
            else:
                try: lmd = lmds[nid]
                except:
                    import pdb
                    pdb.set_trace()
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())
        for i, k in enumerate(self.keyroots):
            self.nodes[k].kidx = i

        # Build a map from node to keyroot
        self.node_to_keyroot = list()
        for i, n in enumerate(self.nodes):
            l = self.lmds[i]
            k = keyroots[l]
            self.node_to_keyroot.append(k)

    def _check_keyroot_map(self):
        # check
        for k_idx in self.keyroots:
            n_idx = k_idx
            while n_idx is not None:
                assert self.node_to_keyroot[n_idx] == k_idx
                ch = self.get_children(self.nodes[n_idx])
                if len(ch) > 0:
                    n_idx = ch[0].nidx
                else:
                    n_idx = None



def simple_distance(A, B, get_children=Node.get_children,
                    get_label=Node.get_label, label_dist=strdist,
                    comparison_filter=None, unique_match_constraints=None,
                    potential_match_fingerprints=None):
    """Computes the exact tree edit distance between trees A and B.

    Use this function if both of these things are true:

    * The cost to insert a node is equivalent to ``label_dist('', new_label)``
    * The cost to remove a node is equivalent to ``label_dist(new_label, '')``

    Otherwise, use :py:func:`zss.distance` instead.

    :param A: The root of a tree.
    :param B: The root of a tree.

    :param get_children:
        A function ``get_children(node) == [node children]``.  Defaults to
        :py:func:`zss.Node.get_children`.

    :param get_label:
        A function ``get_label(node) == 'node label'``.All labels are assumed
        to be strings at this time. Defaults to :py:func:`zss.Node.get_label`.

    :param label_distance:
        A function
        ``label_distance((get_label(node1), get_label(node2)) >= 0``.
        This function should take the output of ``get_label(node)`` and return
        an integer greater or equal to 0 representing how many edits to
        transform the label of ``node1`` into the label of ``node2``. By
        default, this is string edit distance (if available). 0 indicates that
        the labels are the same. A number N represent it takes N changes to
        transform one label into the other.

    :return: An integer distance [0, inf+)
    """
    z = ZSTreeDist(
        A, B, get_children,
        insert_cost=lambda node: label_dist('', get_label(node)),
        remove_cost=lambda node: label_dist(get_label(node), ''),
        update_cost=lambda a, b: label_dist(get_label(a), get_label(b)),
        comparison_filter=comparison_filter,
        unique_match_constraints=unique_match_constraints,
        potential_match_fingerprints=potential_match_fingerprints,
    )
    return z.distance()



class ZSTreeDist (object):
    '''Computes the exact tree edit distance between trees A and B with a
    richer API than :py:func:`zss.simple_distance`.

    Use this function if either of these things are true:

    * The cost to insert a node is **not** equivalent to the cost of changing
      an empty node to have the new node's label
    * The cost to remove a node is **not** equivalent to the cost of changing
      it to a node with an empty label

    Otherwise, use :py:func:`zss.simple_distance`.

    :param A: The root of a tree.
    :param B: The root of a tree.

    :param get_children:
        A function ``get_children(node) == [node children]``.  Defaults to
        :py:func:`zss.Node.get_children`.

    :param insert_cost:
        A function ``insert_cost(node) == cost to insert node >= 0``.

    :param remove_cost:
        A function ``remove_cost(node) == cost to remove node >= 0``.

    :param update_cost:
        A function ``update_cost(a, b) == cost to change a into b >= 0``.

    :return: An integer distance [0, inf+)
    '''
    def __init__(self, A, B, get_children, insert_cost, remove_cost, update_cost,
                 comparison_filter=None, unique_match_constraints=None,
                 potential_match_fingerprints=None):

        self.comparison_filter = comparison_filter

        self.match_a_to_b = {}
        if unique_match_constraints is not None:
            for node_a, node_b in unique_match_constraints:
                self.match_a_to_b[node_a] = node_b
        self.potential_match_fingerprints = potential_match_fingerprints

        self.A, self.B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
        self.treedists = zeros((len(self.A.nodes), len(self.B.nodes)), int)

        self.insert_cost = insert_cost
        self.remove_cost = remove_cost
        self.update_cost = update_cost

        self.comparison_count = 0
        self.filtered_comparison_count = 0
        self.comparisons_filtered_out = 0
        self.comparisons_matched_out = 0

        self.kr_done = zeros((len(self.A.keyroots), len(self.B.keyroots)), int)

        self._cache = {}


    def distance(self):
        # for i in self.A.keyroots:
        #     for j in self.B.keyroots:
        #         self.forest_dist(i, j)

        dist = self.get(len(self.A.nodes)-1, len(self.B.nodes)-1)

        print 'ZS performed {0}/{1} comparisons; {2} saved by filtering, {3} saved by matching'.format(
            self.filtered_comparison_count, self.comparison_count, self.comparisons_filtered_out, self.comparisons_matched_out)

        return dist


    def get(self, i, j):
        k_i = self.A.node_to_keyroot[i]
        k_j = self.B.node_to_keyroot[j]

        kn_i = self.A.nodes[k_i]
        kn_j = self.B.nodes[k_j]

        fg_i = kn_i.fingerprint_index
        fg_j = kn_j.fingerprint_index

        key = fg_i, fg_j

        entry = self._cache.get(key)
        if entry is None:
            self.forest_dist(k_i, k_j)
            entry = k_i, k_j
            self._cache[key] = entry
            return self.treedists[i][j]
        else:
            ck_i, ck_j = entry
            u = k_i - ck_i
            v = k_j - ck_j
            return self.treedists[i-u][j-v]


    def forest_dist(self, i, j):
        Al = self.A.lmds
        Bl = self.B.lmds
        An = self.A.nodes
        Bn = self.B.nodes

        kidx_i = self.A.node_to_keyroot[i]
        kidx_j = self.B.node_to_keyroot[j]
        kn_i = self.A.nodes[kidx_i]
        kn_j = self.B.nodes[kidx_j]

        self.kr_done[kn_i.kidx][kn_j.kidx] = 1

        Ai_fg = An[i].fingerprint_index
        Bj_fg = Bn[j].fingerprint_index

        filter_key = An[i].label, Bn[j].label
        full_test_required = True
        nodes_matched = False
        filtered = False
        match_target = self.match_a_to_b.get(An[i])

        if self.potential_match_fingerprints is not None:
            if Ai_fg in self.potential_match_fingerprints or Bj_fg in self.potential_match_fingerprints:
                full_test_required = False

        if Ai_fg == Bj_fg:
            full_test_required = False
            nodes_matched = True

        if match_target is not None:
            full_test_required = False

            if match_target is Bn[j]:
                nodes_matched = True

        if self.comparison_filter is not None:
            if not self.comparison_filter[filter_key]:
                full_test_required = False
                filtered = True

        # The left-most ancestor of node `i` is `Al[i]`. Its index will be smaller than that of `i`.
        # `i - Al[i] + 1` will be the number of nodes in the subtree rooted at node `i`.
        # For computing edit distance of two sequences of length `r` and `s` a distance matrix of
        # size (r+1, s+1) is required.
        # `m` and `n` are the dimensions of the forest distance matrix `fd`.
        m = i - Al[i] + 2
        n = j - Bl[j] + 2

        # indices into the distance matrix correspond to node_index+1
        # `ioff` and `joff` are the position of the left-most-leaf of the subtrees rooted at `i` and `j`.
        # They are effectively the offset of the local `i` <-> `j` forest distance matrix `fd` within the
        # global distance matrix `treedists`
        # Adding them to an index will transform from forest space to global space
        ioff = Al[i] - 1
        joff = Bl[j] - 1

        if full_test_required:
            fd = zeros((m,n), int)
            for x in xrange(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
                fd[x][0] = fd[x-1][0] + self.remove_cost(An[x+ioff])
            for y in xrange(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
                fd[0][y] = fd[0][y-1] + self.insert_cost(Bn[y+joff])

            # `x` and `y` are indices into the forest distance matrix `fd`
            for x in xrange(1, m): ## the plus one is for the xrange impl
                for y in xrange(1, n):
                    # only need to check if x is an ancestor of i
                    # and y is an ancestor of j
                    if Al[i] == Al[x+ioff] and Bl[j] == Bl[y+joff]:
                        #                   +-
                        #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                        # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                        #                   | δ(l(i1)..i-1, l(j1)..j-1) + γ(v → w)
                        #                   +-
                        fd[x][y] = min(
                            fd[x-1][y] + self.remove_cost(An[x+ioff]),
                            fd[x][y-1] + self.insert_cost(Bn[y+joff]),
                            fd[x-1][y-1] + self.update_cost(An[x+ioff], Bn[y+joff]),
                        )
                        self.treedists[x+ioff][y+joff] = fd[x][y]
                    else:
                        #                   +-
                        #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                        # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                        #                   | δ(l(i1)..l(i)-1, l(j1)..l(j)-1)
                        #                   |                     + treedist(i1,j1)
                        #                   +-

                        # `x+ioff` transforms x from forest space into global space,
                        # `Al[x+ioff]` gets the index of the left most descendant of `x` in global space
                        # `Al[x+ioff] - 1` gets the index of the root node of the previous subtree to `x` in global space
                        # `Al[x+ioff]-1-off` transforms the index of the root of the previous subtree into forest space
                        p = Al[x+ioff]-1-ioff
                        q = Bl[y+joff]-1-joff
                        #print (p, q), (len(fd), len(fd[0]))
                        fd[x][y] = min(
                            fd[x-1][y] + self.remove_cost(An[x+ioff]),
                            fd[x][y-1] + self.insert_cost(Bn[y+joff]),
                            fd[p][q] + self.get(x+ioff, y+joff)
                        )
            self.comparison_count += (m-1) * (n-1)
            self.filtered_comparison_count += (m-1) * (n-1)
        else:
            # Using the normal code above as reference, we can see that
            # we only write the treedist array for nodes that are on the left-most path
            # from the subtree rooted at i,j
            #
            # The nodes on the left most path can be obtained using the code:
            # [x for x in xrange(1, m) if Al[x+ioff] == ll_i]
            #
            # We can however save some array allocation costs by walking the tree
            nx = An[i]
            while nx is not None:
                x = nx.nidx - ioff
                ny = Bn[j]
                while ny is not None:
                    y = ny.nidx - joff
                    if nodes_matched:
                        cost = abs(x-y)
                    else:
                        cost = x + y
                    self.treedists[nx.nidx][ny.nidx] = cost

                    ny = ny.children[0] if len(ny.children) > 0 else None

                nx = nx.children[0] if len(nx.children) > 0 else None

            saved = (m-1) * (n-1)
            self.comparison_count += saved
            if filtered:
                self.comparisons_filtered_out += saved
            else:
                self.comparisons_matched_out += saved
