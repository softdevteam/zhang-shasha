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
from zss.match_list import MatchList


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
        # that is a reversed right-to-left pre-order traversal:
        # The first node is the left-most-leaf, the last is the root

        # `self.nodes` is a list of nodes in order of reversed right-to-left pre-order traversal
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
        # `self.keyroots` is in order of reversed right-to-left pre-order traversal

        lmds = dict()
        keyroots = dict()
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
                    if a not in lmds: lmds[a] = i
                    else: break
            else:
                try: lmd = lmds[nid]
                except:
                    import pdb
                    pdb.set_trace()
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())


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
    def update_cost(a, b):
        if a.fingerprint_index == b.fingerprint_index:
            return 0
        elif a.weight == 1 and b.weight == 1:
            return label_dist(get_label(a), get_label(b))
        else:
            return a.weight + b.weight
    return distance(
        A, B, get_children,
        update_cost=update_cost,
        comparison_filter=comparison_filter,
        unique_match_constraints=unique_match_constraints,
        potential_match_fingerprints=potential_match_fingerprints,
    )


def distance(A, B, get_children, update_cost,
             comparison_filter=None, unique_match_constraints=None,
             potential_match_fingerprints=None):
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
    match_a_to_b = {}
    if unique_match_constraints is not None:
        for node_a, node_b in unique_match_constraints:
            match_a_to_b[node_a] = node_b

    A, B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    treedists = zeros((len(A.nodes), len(B.nodes)), int)
    treematches = [[None for _j in xrange(len(B.nodes))] for _i in xrange(len(A.nodes))]

    comparison_count = [0]
    filtered_comparison_count = [0]
    comparisons_filtered_out = [0]
    comparisons_matched_out = [0]

    def treedist(i, j):
        Al = A.lmds
        Bl = B.lmds
        An = A.nodes
        Bn = B.nodes

        Ai_fg = An[i].fingerprint_index
        Bj_fg = Bn[j].fingerprint_index

        filter_key = An[i].label, Bn[j].label
        full_test_required = True
        nodes_matched = False
        filtered = False
        match_target = match_a_to_b.get(An[i])

        if potential_match_fingerprints is not None:
            if Ai_fg in potential_match_fingerprints or Bj_fg in potential_match_fingerprints:
                full_test_required = False

        if Ai_fg == Bj_fg:
            full_test_required = False
            nodes_matched = True

        if match_target is not None:
            full_test_required = False

            if match_target is Bn[j]:
                nodes_matched = True

        if comparison_filter is not None:
            if not comparison_filter[filter_key]:
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
            fm = [[None for _j in xrange(n)] for _i in xrange(m)]
            for x in xrange(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
                fd[x][0] = fd[x-1][0] + An[x+ioff].weight
            for y in xrange(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
                fd[0][y] = fd[0][y-1] + Bn[y+joff].weight

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
                        del_cost = fd[x-1][y] + An[x+ioff].weight
                        ins_cost = fd[x][y-1] + Bn[y+joff].weight
                        upd_cost = fd[x-1][y-1] + update_cost(An[x+ioff], Bn[y+joff])
                        cost = del_cost
                        mat = fm[x-1][y]
                        if ins_cost < cost:
                            cost = ins_cost
                            mat = fm[x][y-1]
                        if upd_cost <= cost:
                            cost = upd_cost
                            mat = MatchList(x+ioff, y+joff, fm[x-1][y-1])
                        fd[x][y] = cost
                        fm[x][y] = mat

                        treedists[x+ioff][y+joff] = cost
                        treematches[x+ioff][y+joff] = mat
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
                        subforest_xy_cost = treedists[x+ioff][y+joff]
                        subforest_xy_matches = treematches[x+ioff][y+joff]
                        del_cost = fd[x-1][y] + An[x+ioff].weight
                        ins_cost = fd[x][y-1] + Bn[y+joff].weight
                        upd_cost = fd[p][q] + subforest_xy_cost
                        cost = del_cost
                        mat = fm[x-1][y]
                        if ins_cost < cost:
                            cost = ins_cost
                            mat = fm[x][y-1]
                        if upd_cost <= cost:
                            cost = upd_cost
                            # The matches contained in `subforest_xy_matches` will be indices that are relative to
                            # the left most descendant
                            mat = MatchList.join(subforest_xy_matches, fm[p][q])
                        fd[x][y] = cost
                        fm[x][y] = mat
            comparison_count[0] += (m-1) * (n-1)
            filtered_comparison_count[0] += (m-1) * (n-1)
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
                x = nx.node_index - ioff
                ny = Bn[j]
                while ny is not None:
                    y = ny.node_index - joff
                    if nodes_matched:
                        cost = abs(x-y)
                        mat = None
                        for r in xrange(min(x,y)):
                            mat = MatchList(r+1+ioff, r+1+joff, mat)
                    else:
                        cost = x + y
                        mat = None
                    treedists[nx.node_index][ny.node_index] = cost
                    treematches[nx.node_index][ny.node_index] = mat

                    ny = ny.children[0] if len(ny.children) > 0 else None

                nx = nx.children[0] if len(nx.children) > 0 else None

            saved = (m-1) * (n-1)
            comparison_count[0] += saved
            if filtered:
                comparisons_filtered_out[0] += saved
            else:
                comparisons_matched_out[0] += saved


    for i in A.keyroots:
        for j in B.keyroots:
            treedist(i,j)

    print 'ZS performed {0}/{1} comparisons; {2} saved by filtering, {3} saved by matching'.format(
        filtered_comparison_count[0], comparison_count[0], comparisons_filtered_out[0], comparisons_matched_out[0])

    return treedists[-1][-1], treematches[-1][-1]
