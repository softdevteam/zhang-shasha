

class MatchList (object):
    def __init__(self, match_pairs, prev_match=None):
        self.match_pairs = match_pairs
        self.prev_match = prev_match


    def __len__(self):
        prev = 0
        if self.prev_match is not None:
            prev = len(self.prev_match)
        return prev + len(self.match_pairs)



    def offset(self, u, v):
        if self.prev_match is not None:
            prev = self.prev_match.offset(u, v)
        else:
            prev = None
        return MatchList([(a+u, b+v) for a, b in self.match_pairs], prev)


    def matches(self):
        matches = []
        node = self
        while node is not None:
            matches.extend(node.match_pairs)
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
        return MatchList(self.match_pairs, prev)

    @staticmethod
    def join(*xs):
        y = None
        for x in xs:
            if x is not None:
                y = x if y is None else x.prepend(y)
        return y

    @staticmethod
    def as_match_list(x):
        if x is None:
            return []
        else:
            return x.matches()

