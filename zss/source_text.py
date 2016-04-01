import bisect


def longest_common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))

def longest_common_suffix(a, b):
    for i, (x, y) in enumerate(zip(reversed(a), reversed(b))):
        if x != y:
            return i
    return min(len(a), len(b))



class SourceText (object):
    """
    A container class for source text, that supports markers that can be specified by character offset and line/column.
    Stores the text in the original form and as a list of lines.
    """
    def __init__(self, text):
        self.text = text.replace('\r\n', '\n')
        self.lines = text.split('\n')

        self.line_offsets = [0]
        index = 0
        while index is not None:
            try:
                index = index + self.text[index:].index('\n') + 1
            except ValueError:
                index = None
            else:
                self.line_offsets.append(index)

        assert len(self.lines) == len(self.line_offsets)


    def marker_at_pos(self, pos):
        line = bisect.bisect_right(self.line_offsets, pos) - 1
        col = pos - self.line_offsets[line]
        return Marker(self, pos, line, col)

    def marker_at_loc(self, line, col):
        if line == len(self.lines):
            if col == 0:
                return self.marker_at_end()
            else:
                raise ValueError('if line == len(self.lines) then col must be 0, not {0}'.format(col))
        elif line > len(self.lines):
            raise ValueError('line ({0}) must not be greater than len(self.lines) ({1})'.format(line, len(self.lines)))
        pos = self.line_offsets[line] + col
        return Marker(self, pos, line, col)

    def marker_at_start(self):
        return self.marker_at_pos(0)

    def marker_at_end(self):
        return self.marker_at_pos(len(self))

    def marker_at_start_of_line(self, line):
        return self.marker_at_loc(line, 0)

    def marker_at_end_of_line(self, line):
        return self.marker_at_loc(line, len(self.lines[line]))


    def markers_at_end_of_longest_common_prefix(self, src, at_line_boundary=False):
        row = longest_common_prefix(self.lines, src.lines)
        if at_line_boundary:
            return self.marker_at_start_of_line(row), src.marker_at_start_of_line(row)
        else:
            col = longest_common_prefix(self.lines[row], src.lines[row])
            return self.marker_at_loc(row, col), src.marker_at_loc(row, col)

    def markers_at_start_of_longest_common_suffix(self, src, at_line_boundary=False):
        neg_row = longest_common_suffix(self.lines, src.lines)
        if at_line_boundary:
            return self.marker_at_start_of_line(len(self.lines)-neg_row),\
                   src.marker_at_start_of_line(len(src.lines)-neg_row)
        else:
            if neg_row == len(self.lines):
                return self.marker_at_start(), src.marker_at_start_of_line(len(src.lines)-neg_row)
            elif neg_row == len(src.lines):
                return self.marker_at_start_of_line(len(self.lines)-neg_row), src.marker_at_start()
            else:
                a_row = len(self.lines) - neg_row - 1
                b_row = len(src.lines) - neg_row - 1
                a = self.lines[a_row]
                b = src.lines[b_row]
                neg_col = longest_common_suffix(a, b)
                if neg_col == 0:
                    return self.marker_at_start_of_line(len(self.lines)-neg_row),\
                           src.marker_at_start_of_line(len(src.lines)-neg_row)
                else:
                    return self.marker_at_loc(a_row, len(a)-neg_col), src.marker_at_loc(b_row, len(b)-neg_col)


    @classmethod
    def from_file(cls, f):
        return cls(f.read())



    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        if isinstance(item, slice):
            s = slice(Marker.coerce_to_pos(item.start) if item.start is not None else None,
                      Marker.coerce_to_pos(item.stop) if item.stop is not None else None,
                      Marker.coerce_to_pos(item.step) if item.step is not None else None)
            return self.text[s]
        else:
            return self.text[Marker.coerce_to_pos(item)]




class Marker (object):
    """
    Marker identifying a position within a `SourceText`.
    """
    def __init__(self, src_text, pos, line, col):
        self.src_text = src_text
        self.pos = pos
        self.line = line
        self.col = col

    @staticmethod
    def coerce_to_pos(x):
        if isinstance(x, (int, long)):
            return x
        elif isinstance(x, Marker):
            return x.pos
        else:
            raise TypeError('expecting int, long or Marker, not {0}'.format(type(x)))

    def __eq__(self, other):
        if isinstance(other, Marker):
            return self.src_text is other.src_text and \
                    self.pos == other.pos and \
                    self.line == other.line and \
                    self.col == other.col
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, Marker):
            return self.pos < other.pos
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, Marker):
            return self.pos <= other.pos
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Marker):
            return self.pos > other.pos
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Marker):
            return self.pos >= other.pos
        else:
            return NotImplemented

    def __cmp__(self, other):
        if isinstance(other, Marker):
            return cmp(self.pos, other.pos)
        else:
            return NotImplemented

    def __str__(self):
        return 'Marker(id(src)={0}, pos={1}, line={2}, col={3})'.format(id(self.src_text.text), self.pos, self.line, self.col)

    def __repr__(self):
        return 'Marker(id(src)={0}, pos={1}, line={2}, col={3})'.format(id(self.src_text.text), self.pos, self.line, self.col)


import unittest

class Test_SourceText (unittest.TestCase):
    def test_longest_common_prefix_and_suffix(self):
        a = [1, 2, 3, 4, 11, 12, 13]
        b = [1, 2, 3, 7, 8, 8, 9, 12, 13]
        self.assertEqual(longest_common_prefix(a, b), 3)
        self.assertEqual(longest_common_suffix(a, b), 2)

        a = [1, 2, 3, 4]
        b = [1, 2, 3, 4]
        self.assertEqual(longest_common_prefix(a, b), 4)
        self.assertEqual(longest_common_suffix(a, b), 4)

        a = [1, 2, 3, 4, 5, 6]
        b = [1, 2, 3, 4]
        self.assertEqual(longest_common_prefix(a, b), 4)
        self.assertEqual(longest_common_suffix(a, b), 0)

        a = [1, 2, 3, 4]
        b = [1, 2, 3, 4, 5, 6]
        self.assertEqual(longest_common_prefix(a, b), 4)
        self.assertEqual(longest_common_suffix(a, b), 0)

        a = [-1, 0, 1, 2, 3, 4]
        b = [1, 2, 3, 4]
        self.assertEqual(longest_common_prefix(a, b), 0)
        self.assertEqual(longest_common_suffix(a, b), 4)

        a = [1, 2, 3, 4]
        b = [-1, 0, 1, 2, 3, 4]
        self.assertEqual(longest_common_prefix(a, b), 0)
        self.assertEqual(longest_common_suffix(a, b), 4)

    def test_construct(self):
        a = SourceText('hello')
        self.assertEqual(a.text, 'hello')
        self.assertEqual(a.lines, ['hello'])
        self.assertEqual(a.line_offsets, [0])
        self.assertEqual(len(a), 5)

        a = SourceText('hello\n')
        self.assertEqual(a.text, 'hello\n')
        self.assertEqual(a.lines, ['hello', ''])
        self.assertEqual(a.line_offsets, [0, 6])
        self.assertEqual(len(a), 6)

        a = SourceText('\nhello')
        self.assertEqual(a.text, '\nhello')
        self.assertEqual(a.lines, ['', 'hello'])
        self.assertEqual(a.line_offsets, [0, 1])
        self.assertEqual(len(a), 6)

        a = SourceText('hello\n world')
        self.assertEqual(a.text, 'hello\n world')
        self.assertEqual(a.lines, ['hello', ' world'])
        self.assertEqual(a.line_offsets, [0, 6])
        self.assertEqual(len(a), 12)

        
    def test_getitem(self):
        a = SourceText('hello\n world\nthis is the end')
        self.assertEqual(a[3], 'l')
        self.assertEqual(a[3:5], 'lo')
        self.assertEqual(a[3:10], 'lo\n wor')
        self.assertEqual(a[a.marker_at_pos(3)], 'l')
        self.assertEqual(a[a.marker_at_loc(0, 3)], 'l')
        self.assertEqual(a[a.marker_at_pos(3):a.marker_at_pos(5)], 'lo')
        self.assertEqual(a[a.marker_at_loc(0, 3):a.marker_at_loc(0, 5)], 'lo')
        self.assertEqual(a[a.marker_at_pos(16):a.marker_at_pos(20)], 's is')
        self.assertEqual(a[a.marker_at_loc(2, 3):a.marker_at_loc(2, 7)], 's is')


    def test_marker_at_start_and_end(self):
        a = SourceText('hello\n world\nthis is the end')
        self.assertEqual(a.marker_at_start().pos, 0)
        self.assertEqual(a.marker_at_end().pos, len(a))


    def test_marker_at_pos(self):
        a = SourceText('hello\n world\nthis is the end')
        self.assertEqual(a.marker_at_pos(0), Marker(a, 0, 0, 0))
        self.assertEqual(a.marker_at_pos(1), Marker(a, 1, 0, 1))
        self.assertEqual(a.marker_at_pos(2), Marker(a, 2, 0, 2))
        self.assertEqual(a.marker_at_pos(3), Marker(a, 3, 0, 3))
        self.assertEqual(a.marker_at_pos(4), Marker(a, 4, 0, 4))
        self.assertEqual(a.marker_at_pos(5), Marker(a, 5, 0, 5))
        self.assertEqual(a.marker_at_pos(6), Marker(a, 6, 1, 0))
        self.assertEqual(a.marker_at_pos(7), Marker(a, 7, 1, 1))
        self.assertEqual(a.marker_at_pos(8), Marker(a, 8, 1, 2))
        self.assertEqual(a.marker_at_pos(9), Marker(a, 9, 1, 3))
        self.assertEqual(a.marker_at_pos(10), Marker(a, 10, 1, 4))
        self.assertEqual(a.marker_at_pos(11), Marker(a, 11, 1, 5))
        self.assertEqual(a.marker_at_pos(12), Marker(a, 12, 1, 6))
        self.assertEqual(a.marker_at_pos(13), Marker(a, 13, 2, 0))
        self.assertEqual(a.marker_at_pos(14), Marker(a, 14, 2, 1))
        self.assertEqual(a.marker_at_pos(15), Marker(a, 15, 2, 2))
        self.assertEqual(a.marker_at_pos(16), Marker(a, 16, 2, 3))
        self.assertEqual(a.marker_at_pos(17), Marker(a, 17, 2, 4))
        self.assertEqual(a.marker_at_pos(18), Marker(a, 18, 2, 5))
        self.assertEqual(a.marker_at_pos(19), Marker(a, 19, 2, 6))
        self.assertEqual(a.marker_at_pos(20), Marker(a, 20, 2, 7))
        self.assertEqual(a.marker_at_pos(21), Marker(a, 21, 2, 8))
        self.assertEqual(a.marker_at_pos(22), Marker(a, 22, 2, 9))
        self.assertEqual(a.marker_at_pos(23), Marker(a, 23, 2, 10))
        self.assertEqual(a.marker_at_pos(24), Marker(a, 24, 2, 11))
        self.assertEqual(a.marker_at_pos(25), Marker(a, 25, 2, 12))
        self.assertEqual(a.marker_at_pos(26), Marker(a, 26, 2, 13))
        self.assertEqual(a.marker_at_pos(27), Marker(a, 27, 2, 14))
        self.assertEqual(a.marker_at_pos(28), Marker(a, 28, 2, 15))


    def test_marker_at_loc(self):
        a = SourceText('hello\n world\nthis is the end')
        self.assertEqual(a.marker_at_loc(0, 0), Marker(a, 0, 0, 0))
        self.assertEqual(a.marker_at_loc(0, 1), Marker(a, 1, 0, 1))
        self.assertEqual(a.marker_at_loc(0, 2), Marker(a, 2, 0, 2))
        self.assertEqual(a.marker_at_loc(0, 3), Marker(a, 3, 0, 3))
        self.assertEqual(a.marker_at_loc(0, 4), Marker(a, 4, 0, 4))
        self.assertEqual(a.marker_at_loc(0, 5), Marker(a, 5, 0, 5))
        self.assertEqual(a.marker_at_loc(1, 0), Marker(a, 6, 1, 0))
        self.assertEqual(a.marker_at_loc(1, 1), Marker(a, 7, 1, 1))
        self.assertEqual(a.marker_at_loc(1, 2), Marker(a, 8, 1, 2))
        self.assertEqual(a.marker_at_loc(1, 3), Marker(a, 9, 1, 3))
        self.assertEqual(a.marker_at_loc(1, 4), Marker(a, 10, 1, 4))
        self.assertEqual(a.marker_at_loc(1, 5), Marker(a, 11, 1, 5))
        self.assertEqual(a.marker_at_loc(1, 6), Marker(a, 12, 1, 6))
        self.assertEqual(a.marker_at_loc(2, 0), Marker(a, 13, 2, 0))
        self.assertEqual(a.marker_at_loc(2, 1), Marker(a, 14, 2, 1))
        self.assertEqual(a.marker_at_loc(2, 2), Marker(a, 15, 2, 2))
        self.assertEqual(a.marker_at_loc(2, 3), Marker(a, 16, 2, 3))
        self.assertEqual(a.marker_at_loc(2, 4), Marker(a, 17, 2, 4))
        self.assertEqual(a.marker_at_loc(2, 5), Marker(a, 18, 2, 5))
        self.assertEqual(a.marker_at_loc(2, 6), Marker(a, 19, 2, 6))
        self.assertEqual(a.marker_at_loc(2, 7), Marker(a, 20, 2, 7))
        self.assertEqual(a.marker_at_loc(2, 8), Marker(a, 21, 2, 8))
        self.assertEqual(a.marker_at_loc(2, 9), Marker(a, 22, 2, 9))
        self.assertEqual(a.marker_at_loc(2, 10), Marker(a, 23, 2, 10))
        self.assertEqual(a.marker_at_loc(2, 11), Marker(a, 24, 2, 11))
        self.assertEqual(a.marker_at_loc(2, 12), Marker(a, 25, 2, 12))
        self.assertEqual(a.marker_at_loc(2, 13), Marker(a, 26, 2, 13))
        self.assertEqual(a.marker_at_loc(2, 14), Marker(a, 27, 2, 14))
        self.assertEqual(a.marker_at_loc(2, 15), Marker(a, 28, 2, 15))


    def test_common_prefix_markers(self):
        a = SourceText('hello\n'
                       'world\n'
                       'this\n'
                       'is the\n'
                       'end')
        b = SourceText('hello\n'
                       'there\n'
                       'is\n'
                       'this\n'
                       'the\n'
                       'end')
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=True),
                         (a.marker_at_start_of_line(1), b.marker_at_start_of_line(1)))
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=False),
                         (a.marker_at_start_of_line(1), b.marker_at_start_of_line(1)))

        b = SourceText('hello\n'
                       'world, now\n'
                       'is\n'
                       'this\n'
                       'the\n'
                       'end')
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=True),
                         (a.marker_at_start_of_line(1), b.marker_at_start_of_line(1)))
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=False),
                         (a.marker_at_loc(1, 5), b.marker_at_loc(1, 5)))

        b = SourceText('hello\n'
                       'word, now\n'
                       'is\n'
                       'this\n'
                       'the\n'
                       'end')
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=True),
                         (a.marker_at_start_of_line(1), b.marker_at_start_of_line(1)))
        self.assertEqual(a.markers_at_end_of_longest_common_prefix(b, at_line_boundary=False),
                         (a.marker_at_loc(1, 3), b.marker_at_loc(1, 3)))


    def test_common_suffix_markers(self):
        a = SourceText('hello\n'
                       'world\n'
                       'this\n'
                       'is the\n'
                       'end')
        b = SourceText('hello\n'
                       'there\n'
                       'that\n'
                       'is the\n'
                       'end')
        self.assertEqual(a.markers_at_start_of_longest_common_suffix(b, at_line_boundary=True),
                         (a.marker_at_start_of_line(3), b.marker_at_start_of_line(3)))
        self.assertEqual(a.markers_at_start_of_longest_common_suffix(b, at_line_boundary=False),
                         (a.marker_at_start_of_line(3), b.marker_at_start_of_line(3)))

        b = SourceText('hello\n'
                       'there\n'
                       'is\n'
                       'this\n'
                       'the\n'
                       'end')
        self.assertEqual(a.markers_at_start_of_longest_common_suffix(b, at_line_boundary=True),
                         (a.marker_at_start_of_line(4), b.marker_at_start_of_line(5)))
        self.assertEqual(a.markers_at_start_of_longest_common_suffix(b, at_line_boundary=False),
                         (a.marker_at_loc(3, 3), b.marker_at_loc(4, 0)))
