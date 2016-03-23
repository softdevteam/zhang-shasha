

class MatchList (object):
    def __init__(self, node_a_index, node_b_index, prev_match=None):
        self.node_a_index = node_a_index
        self.node_b_index = node_b_index
        self.prev_match = prev_match


    def with_matcb(self, node_a_index, node_b_index):
        return MatchList(node_a_index, node_b_index, self)

    def offset(self, u, v):
        if self.prev_match is not None:
            prev = self.prev_match.offset(u, v)
        else:
            prev = None
        return MatchList(self.node_a_index + u, self.node_b_index + v, prev)


    def matches(self):
        matches = []
        node = self
        while node is not None:
            matches.append((node.node_a_index, node.node_b_index))
            node = node.prev_match

        sorted_matches = sorted(matches)
        for i, pair in enumerate(sorted_matches):
            # print pair
            for ab in sorted_matches[:i]:
                if ab == pair:
                    print 'Warning: match {0}->{1} appears more than once'.format(pair[0], pair[1])
                if ab[0] == pair[0] and ab[1] != pair[1]:
                    print 'Error: match {0}->{1} conflicts with {2}->{3}'.format(pair[0], pair[1], ab[0], ab[1])

        return matches

    def prepend(self, prefix):
        if self.prev_match is not None:
            prev = self.prev_match.prepend(prefix)
        else:
            prev = prefix
        return MatchList(self.node_a_index, self.node_b_index, prev)

    @staticmethod
    def join(a, b):
        if a is not None and b is not None:
            return b.prepend(a)
        else:
            return a or b

    @staticmethod
    def as_match_list(x):
        if x is None:
            return []
        else:
            return x.matches()

