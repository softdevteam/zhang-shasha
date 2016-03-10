def lcs_matches(a, b):
    """
    Compute longest common sub-sequence of two sequences

    :param a: The first sequence
    :param b: The second sequence
    :return: `(lcs, matches)` where `lcs` is the length of the LCS and `matches` is a list of tuples
    `[(i0, j0), (i1, j1), ... (in, jn)]` where `ix` and `jx` are indices of matching elements from `a` and `b`
    respectively.
    """
    OP_DEL = 1
    OP_INS = 2
    OP_UPD = 4

    n = len(a)
    m = len(b)
    p = n+1
    q = m+1

    score_matrix = [0] * p * q
    op_matrix = [0] * p * q

    # Top row - deletes
    for i in xrange(1, p):
        score_matrix[0*p + i] = 0
        op_matrix[0*p + i] = op_matrix[0*p + (i-1)] | OP_DEL
    # Left column - inserts
    for j in xrange(1, q):
        score_matrix[j*p + 0] = 0
        op_matrix[j*p + 0] = op_matrix[(j-1)*p + 0] | OP_INS

    # Main body
    for j in xrange(1, q):
        for i in xrange(1, p):
            local_score = 1 if a[i-1] == b[j-1] else 0
            del_score = score_matrix[j*p + (i-1)]
            ins_score = score_matrix[(j-1)*p + i]
            upd_score = score_matrix[(j-1)*p + (i-1)] + local_score
            score = max(del_score, ins_score, upd_score)
            score_matrix[j*p + i] = score

            op = 0
            if del_score == score and local_score == 0:
                op |= OP_DEL
            elif ins_score == score and local_score == 0:
                op |= OP_INS
            elif upd_score == score:
                op |= OP_UPD
            op_matrix[j*p + i] = op

    lcs = score_matrix[m*p+n]
    i = n
    j = m
    matches = [None] * lcs
    pos = lcs - 1
    while i > 0 and j > 0:
        if (op_matrix[j*p + i] & OP_UPD) != 0:
            matches[pos] = (i-1, j-1)
            pos -= 1
            i -= 1
            j -= 1
        elif (op_matrix[j*p + i] & OP_DEL) != 0:
            i -= 1
        elif (op_matrix[j*p + i] & OP_INS) != 0:
            j -= 1
        else:
            raise ValueError('No operation at position {0},{1}'.format(i, j))

    return lcs, matches


import unittest

class Test_lcs (unittest.TestCase):
    def test_lcs_matches(self):
        self.assertEqual(lcs_matches('hello', 'hell'), (4, [(0,0), (1,1), (2,2), (3,3)]))
        self.assertEqual(lcs_matches('hello', 'llo'), (3, [(2,0), (3,1), (4,2)]))
        self.assertEqual(lcs_matches('hello world', 'go away'), (3, [(4,1), (5,2), (6,4)]))