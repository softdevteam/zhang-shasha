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
        self.keyroot_indices = None
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

        # `self.keyroots` is also a list of indices that index into `self.nodes`
        # It lists the indices of keyroot nodes
        # For each keyroot node `k` there exists no keyroot node `j` that has the same left-most-descendant
        # as `k` where `j` is a descendant of `k`, in other words, each `k` in `self.keyroots` has
        # a different left-most-descendant, and `k` is the furthest ancestor from
        # `left_most_descendant(k)`
        # `self.keyroots` is in order of left-to-right post-order traversal

        node_ndx_to_lmd_ndx = dict()
        lmd_ndx_to_keyroot_ndx = dict()
        i = 0
        while len(pstack) > 0:
            (n, nid), anc = pstack.pop()
            #print list(anc)
            n.node_index = len(self.nodes)
            self.nodes.append(n)
            self.ids.append(nid)
            #print n.label, [a.label for a in anc]
            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in node_ndx_to_lmd_ndx:
                        node_ndx_to_lmd_ndx[a] = i
                    else:
                        break
            else:
                try:
                    lmd = node_ndx_to_lmd_ndx[nid]
                except:
                    import pdb
                    pdb.set_trace()
            n.left_most_descendant_index = lmd
            lmd_ndx_to_keyroot_ndx[lmd] = i
            i += 1
        self.keyroot_indices = sorted(lmd_ndx_to_keyroot_ndx.values())

        # Build a map from node to keyroot
        for i, n in enumerate(self.nodes):
            lmd_ndx = n.left_most_descendant_index
            k_ndx = lmd_ndx_to_keyroot_ndx[lmd_ndx]
            keyroot = self.nodes[k_ndx]
            dist_to_keyroot = 0
            while keyroot is not n:
                dist_to_keyroot += 1
                keyroot = keyroot.children[0]
            n.keyroot_node_index = k_ndx
            n.dist_to_keyroot = dist_to_keyroot

        for i, k in enumerate(self.keyroot_indices):
            keyroot = self.nodes[k]
            keyroot.index_in_keyroot_list = i
            node = keyroot
            dist = 0
            while node is not None:
                dist += 1
                node = node.children[0] if len(node.children) > 0 else None
            keyroot.keyroot_path_length = dist



def simple_distance(A, B, N_fingerprints, get_children=Node.get_children,
                    get_label=Node.get_label, label_dist=strdist,
                    comparison_filter=None, unique_match_constraints=None,
                    potential_match_fingerprints=None, verbose=False):
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
    def update_cost(a, b):
        if a.fingerprint_index == b.fingerprint_index:
            return 0
        elif a.weight == 1 and b.weight == 1:
            return label_dist(get_label(a), get_label(b))
        else:
            return a.weight + b.weight
    z = ZSTreeDist(
        A, B, N_fingerprints, get_children,
        update_cost=update_cost,
        comparison_filter=comparison_filter,
        unique_match_constraints=unique_match_constraints,
        potential_match_fingerprints=potential_match_fingerprints,
    )
    return z.distance(verbose=verbose)



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
    OP_UPD = 0x1
    OP_JOIN = 0x2
    OP_DEL = 0x4
    OP_INS = 0x8


    def __init__(self, A, B, N_fingerprints, get_children, update_cost,
                 comparison_filter=None, unique_match_constraints=None,
                 potential_match_fingerprints=None):

        self.comparison_filter = comparison_filter

        self.match_a_to_b = {}
        if unique_match_constraints is not None:
            for node_a, node_b in unique_match_constraints:
                self.match_a_to_b[node_a] = node_b
        self.potential_match_fingerprints = potential_match_fingerprints

        self.A, self.B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
        self.N_fingerprints = N_fingerprints

        self.update_cost = update_cost

        self.comparison_count = 0
        self.filtered_comparison_count = 0
        self.comparisons_filtered_out = 0
        self.comparisons_matched_out = 0

        self._cache = [None] * (N_fingerprints * N_fingerprints)


    def distance(self, verbose=False):
        # for i in self.A.keyroots:
        #     for j in self.B.keyroots:
        #         self.forest_dist(i, j)

        matches = []
        subforests_for_matching = []

        dist = self.get(matches, subforests_for_matching, len(self.A.nodes)-1, len(self.B.nodes)-1)

        # We redo some of our matching work here; just the subforests on the critical path, this time
        # filling in the list of matches as we go
        while len(subforests_for_matching) > 0:
            a, b = subforests_for_matching.pop()
            self.forest_dist(None, matches, subforests_for_matching, a, b)

        if verbose:
            print 'ZS performed {0}/{1} comparisons; {2} saved by filtering, {3} saved by matching'.format(
                self.filtered_comparison_count, self.comparison_count, self.comparisons_filtered_out, self.comparisons_matched_out)

        node_matches = [(self.A.nodes[i], self.B.nodes[j]) for i, j in matches]

        return dist, node_matches


    def get(self, matches, subforests_for_matching, i, j):
        Ai = self.A.nodes[i]
        Bj = self.B.nodes[j]

        k_i = Ai.keyroot_node_index
        k_j = Bj.keyroot_node_index

        kn_i = self.A.nodes[k_i]
        kn_j = self.B.nodes[k_j]

        fg_i = kn_i.fingerprint_index
        fg_j = kn_j.fingerprint_index

        cache_index = fg_i * self.N_fingerprints + fg_j

        left_path_table = self._cache[cache_index]
        if left_path_table is None:
            P = kn_i.keyroot_path_length
            Q = kn_j.keyroot_path_length
            left_path_table = zeros((P, Q), int)
            self._cache[cache_index] = left_path_table
            self.forest_dist(left_path_table, matches, subforests_for_matching, k_i, k_j)
            return left_path_table[0][0]
        else:
            p = Ai.dist_to_keyroot
            q = Bj.dist_to_keyroot
            return left_path_table[p][q]


    def forest_dist(self, left_path_table, matches, subforests_for_matching, i, j):
        An = self.A.nodes
        Bn = self.B.nodes

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
        m = i - An[i].left_most_descendant_index + 2
        n = j - Bn[j].left_most_descendant_index + 2

        # indices into the distance matrix correspond to node_index+1
        # `ioff` and `joff` are the position of the left-most-leaf of the subtrees rooted at `i` and `j`.
        # They are effectively the offset of the local `i` <-> `j` forest distance matrix `fd` within the
        # global distance matrix `treedists`
        # Adding them to an index will transform from forest space to global space
        ioff = An[i].left_most_descendant_index - 1
        joff = Bn[j].left_most_descendant_index - 1

        if full_test_required:
            fd = zeros((m,n), int)
            if matches is not None:
                fo = zeros((m,n), int)
            else:
                fo = None
            for x in xrange(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
                fd[x][0] = fd[x-1][0] + An[x+ioff].weight
                if matches is not None:
                    fo[x][0] = self.OP_DEL
            for y in xrange(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
                fd[0][y] = fd[0][y-1] + Bn[y+joff].weight
                if matches is not None:
                    fo[0][y] = self.OP_INS

            # `x` and `y` are indices into the forest distance matrix `fd`
            for x in xrange(1, m): ## the plus one is for the xrange impl
                for y in xrange(1, n):
                    # only need to check if x is an ancestor of i
                    # and y is an ancestor of j
                    if An[i].left_most_descendant_index == An[x+ioff].left_most_descendant_index and \
                                    Bn[j].left_most_descendant_index == Bn[y+joff].left_most_descendant_index:
                        #                   +-
                        #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                        # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                        #                   | δ(l(i1)..i-1, l(j1)..j-1) + γ(v → w)
                        #                   +-
                        del_cost_op = fd[x-1][y] + An[x+ioff].weight, self.OP_DEL
                        ins_cost_op = fd[x][y-1] + Bn[y+joff].weight, self.OP_INS
                        upd_cost_op = fd[x-1][y-1] + self.update_cost(An[x+ioff], Bn[y+joff]), self.OP_UPD
                        cost, op = min(upd_cost_op, del_cost_op, ins_cost_op)
                        fd[x][y] = cost

                        if matches is not None:
                            fo[x][y] = op
                        if left_path_table is not None:
                            left_path_table[An[x + ioff].dist_to_keyroot][Bn[y + joff].dist_to_keyroot] = cost
                    else:
                        #                   +-
                        #                   | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
                        # δ(F1 , F2 ) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
                        #                   | δ(l(i1)..l(i)-1, l(j1)..l(j)-1)
                        #                   |                     + treedist(i1,j1)
                        #                   +-

                        # `x+ioff` transforms x from forest space into global space,
                        # `An[x+ioff].left_most_descendant_index` gets the index of the left most descendant of `x`
                        # in global space
                        # `An[x+ioff].left_most_descendant_index - 1` gets the index of the root node of the
                        # previous subtree to `x` in global space
                        # `An[x+ioff].left_most_descendant_index - 1 - ioff` transforms the index of the root of
                        # the previous subtree into forest space
                        # Therefore `p` is the index of the root of the previous subtree in forest space
                        p = An[x+ioff].left_most_descendant_index-1-ioff
                        q = Bn[y+joff].left_most_descendant_index-1-joff
                        #print (p, q), (len(fd), len(fd[0]))
                        subforest_xy_cost = self.get(None, None, x+ioff, y+joff)
                        del_cost_op = fd[x-1][y] + An[x+ioff].weight, self.OP_DEL
                        ins_cost_op = fd[x][y-1] + Bn[y+joff].weight, self.OP_INS
                        join_cost_op = fd[p][q] + subforest_xy_cost, self.OP_JOIN
                        cost, op = min(join_cost_op, del_cost_op, ins_cost_op)
                        fd[x][y] = cost
                        if matches is not None:
                            fo[x][y] = op
            self.comparison_count += (m-1) * (n-1)
            self.filtered_comparison_count += (m-1) * (n-1)

            if matches is not None:
                u = m - 1
                v = n - 1
                while u > 0 or v > 0:
                    opcode = fo[u][v]
                    if opcode == self.OP_JOIN:
                        subforests_for_matching.append((u+ioff, v+joff))
                        u = An[u+ioff].left_most_descendant_index - 1 - ioff
                        v = Bn[v+joff].left_most_descendant_index - 1 - joff
                    elif (opcode & self.OP_UPD) != 0:
                        matches.append((u+ioff, v+joff))
                        u -= 1
                        v -= 1
                    elif (opcode & self.OP_DEL) != 0:
                        u -= 1
                    elif (opcode & self.OP_INS) != 0:
                        v -= 1
                    else:
                        raise ValueError('Unknown op code {0}'.format(fo[u][v]))
        else:
            # Using the normal code above as reference, we can see that
            # we only write the treedist array for nodes that are on the left-most path
            # from the subtree rooted at i,j
            #
            # The nodes on the left most path can be obtained using the code:
            # [x for x in xrange(1, m) if Al[x+ioff] == ll_i]
            #
            # We can however save some array allocation costs by walking the tree
            if left_path_table is not None:
                nx = An[i]
                while nx is not None:
                    x = nx.node_index - ioff
                    ny = Bn[j]
                    while ny is not None:
                        y = ny.node_index - joff
                        if nodes_matched:
                            cost = abs(x-y)
                        else:
                            cost = x + y
                        # self.treedists[nx.node_index][ny.node_index] = cost
                        left_path_table[nx.dist_to_keyroot][ny.dist_to_keyroot] = cost

                        ny = ny.children[0] if len(ny.children) > 0 else None

                    nx = nx.children[0] if len(nx.children) > 0 else None

            if matches is not None:
                for r in xrange(1, min(m, n)):
                    matches.append((r+ioff, r+joff))

            saved = (m-1) * (n-1)
            self.comparison_count += saved
            if filtered:
                self.comparisons_filtered_out += saved
            else:
                self.comparisons_matched_out += saved
