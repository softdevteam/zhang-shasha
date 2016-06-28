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

    :param comparison_filter:
        [optional] a dictionary that determines if two nodes can possibly match by their
        type labels, of the form `{(label_a, label_b): can_match}` where
        `label_a` is the type label from node A, `label_b` is the type label
        from node B and `can_match` is a boolean indicating if two nodes
        with these type labels can be matched

    :param unique_match_constraints:
        [optional] A list of node pairs of the form `[(node_from_A, node_from_B), ...]`
        that list nodes that are matched to one another as a result of
        the subtrees rooted at them being found to be equal by a pre-processing
        step (see `fg_match` module)

    :param potential_match_fingerprints:
        [optional] A set of node fingerprint indices that are used by the
        same number of nodes in both trees. Since they are used more than
        once in each tree they cannot be trivially matched; e.g. if there are
        two nodes x0, and x1 in tree A and two nodes y0 and y1 that use
        fingerprint i, x0 could match to either y0 or y1. We do however know
        that x0 will match to either y0 or y1, but no other node. This can
        be used to speed up the matching process by not fully exploring matches
        that involve nodes whose fingerprints are in this set.

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
        verbose=verbose,
    )


OP_UPD = 0x1
OP_JOIN = 0x2
OP_DEL = 0x4
OP_INS = 0x8


def distance(A, B, get_children, update_cost,
             comparison_filter=None, unique_match_constraints=None,
             potential_match_fingerprints=None, verbose=False):
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

    :param comparison_filter:
        [optional] a dictionary that determines if two nodes can possibly match by their
        type labels, of the form `{(label_a, label_b): can_match}` where
        `label_a` is the type label from node A, `label_b` is the type label
        from node B and `can_match` is a boolean indicating if two nodes
        with these type labels can be matched

    :param unique_match_constraints:
        [optional] A list of node pairs of the form `[(node_from_A, node_from_B), ...]`
        that list nodes that are matched to one another as a result of
        the subtrees rooted at them being found to be equal by a pre-processing
        step (see `fg_match` module)

    :param potential_match_fingerprints:
        [optional] A set of node fingerprint indices that are used by the
        same number of nodes in both trees. Since they are used more than
        once in each tree they cannot be trivially matched; e.g. if there are
        two nodes x0, and x1 in tree A and two nodes y0 and y1 that use
        fingerprint i, x0 could match to either y0 or y1. We do however know
        that x0 will match to either y0 or y1, but no other node. This can
        be used to speed up the matching process by not fully exploring matches
        that involve nodes whose fingerprints are in this set.

    :return: An integer distance [0, inf+)
    '''
    match_a_to_b = {}
    if unique_match_constraints is not None:
        for node_a, node_b in unique_match_constraints:
            match_a_to_b[node_a] = node_b

    A, B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    treedists = zeros((len(A.nodes), len(B.nodes)), int)

    comparison_count = [0]
    filtered_comparison_count = [0]
    comparisons_filtered_out = [0]
    comparisons_matched_out = [0]

    def treedist(i, j, write_to_treedists, matches, subforests_for_matching):
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
            if matches is not None:
                fo = zeros((m,n), int)
            else:
                fo = None

            for x in xrange(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
                fd[x][0] = fd[x-1][0] + An[x+ioff].weight
                if matches is not None:
                    fo[x][0] = OP_DEL
            for y in xrange(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
                fd[0][y] = fd[0][y-1] + Bn[y+joff].weight
                if matches is not None:
                    fo[0][y] = OP_INS

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
                        del_cost_op = fd[x-1][y] + An[x+ioff].weight, OP_DEL
                        ins_cost_op = fd[x][y-1] + Bn[y+joff].weight, OP_INS
                        upd_cost_op = fd[x-1][y-1] + update_cost(An[x+ioff], Bn[y+joff]), OP_UPD
                        cost, op = min(upd_cost_op, del_cost_op, ins_cost_op)
                        fd[x][y] = cost
                        if matches is not None:
                            fo[x][y] = op
                        if write_to_treedists:
                            treedists[x+ioff][y+joff] = cost
                        # print '{0},{1} full: written {2}'.format(x+ioff, y+joff, len(treematches[x+ioff][y+joff]))
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
                        del_cost_op = fd[x-1][y] + An[x+ioff].weight, OP_DEL
                        ins_cost_op = fd[x][y-1] + Bn[y+joff].weight, OP_INS
                        join_cost_op = fd[p][q] + subforest_xy_cost, OP_JOIN
                        cost, op = min(join_cost_op, del_cost_op, ins_cost_op)
                        fd[x][y] = cost
                        if matches is not None:
                            fo[x][y] = op

            if matches is not None:
                u = m - 1
                v = n - 1
                while u > 0 or v > 0:
                    opcode = fo[u][v]
                    if opcode == OP_JOIN:
                        subforests_for_matching.append((u+ioff, v+joff))
                        u = Al[u+ioff]-1-ioff
                        v = Bl[v+joff]-1-joff
                    elif (opcode & OP_UPD) != 0:
                        matches.append((u+ioff, v+joff))
                        u -= 1
                        v -= 1
                    elif (opcode & OP_DEL) != 0:
                        u -= 1
                    elif (opcode & OP_INS) != 0:
                        v -= 1
                    else:
                        raise ValueError('Unknown op code {0}'.format(fo[u][v]))


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
            if write_to_treedists:
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
                        treedists[nx.node_index][ny.node_index] = cost

                        ny = ny.children[0] if len(ny.children) > 0 else None

                    nx = nx.children[0] if len(nx.children) > 0 else None

            if matches is not None:
                for r in xrange(1, min(m, n)):
                    matches.append((r+ioff, r+joff))

            saved = (m-1) * (n-1)
            comparison_count[0] += saved
            if filtered:
                comparisons_filtered_out[0] += saved
            else:
                comparisons_matched_out[0] += saved


    matches = []
    subforests_for_matching = []
    # Match keyroots with one another
    for i in A.keyroots:
        for j in B.keyroots:
            if i == A.keyroots[-1] and j == B.keyroots[-1]:
                # Root node; this is the last pair of subforests so write to `treedists` and provide
                # lists of matches and further subforests to fill in
                treedist(i, j, True, matches, subforests_for_matching)
            else:
                treedist(i, j, True, None, None)

    # We redo some of our matching work here; just the subforests on the critical path, this time
    # filling in the list of matches as we go
    while len(subforests_for_matching) > 0:
        a, b = subforests_for_matching.pop()
        treedist(a, b, False, matches, subforests_for_matching)


    if verbose:
        print 'ZS performed {0}/{1} comparisons; {2} saved by filtering, {3} saved by matching'.format(
            filtered_comparison_count[0], comparison_count[0], comparisons_filtered_out[0], comparisons_matched_out[0])

    node_matches = [(A.nodes[i], B.nodes[j]) for i, j in matches]

    return treedists[-1][-1], node_matches

