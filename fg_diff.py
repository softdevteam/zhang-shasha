import argparse, parser, time

from ast_to_simple_tree import ASTConverter
import example_data
from zss import fg_match


def test_fg_match(A_src, B_src):
    conv = ASTConverter()

    A = conv.parse(A_src)
    B = conv.parse(B_src)


    print('SOURCE CODE STATS: |A_src|={0}, |B_src|={1}, |A_src.lines|={2}, |B_src.lines|={3}'.format(
        len(A_src), len(B_src), len(A_src.lines), len(B_src.lines)))


    t1 = time.time()

    fv = fg_match.FeatureVectorTable()
    fv.add_tree(A)
    fv.add_tree(B)



    print('BASE CASE TREE STATS: |A|={0}, |B|={1}, A.height={2}, B.height={3}, ' \
                             '|shape_fingerprints(A, B)|={4}'.format(
        len([x for x in A.iter()]), len([x for x in B.iter()]),
        A.depth, B.depth, len(fv.fingerprints)))

    t2 = time.time()

    top_down_matches = fg_match.top_down_match_nodes_by_fingerprint(A, B)
    n_top_down_matched_nodes = 0
    for a, b in top_down_matches:
        n_top_down_matched_nodes += a.subtree_size

    t3 = time.time()

    bottom_up_matches = fg_match.greedy_bottom_up_match_nodes_by_fingerprint(A, B)

    t4 = time.time()

    print 'Fingerprint generation: {:.2f}s, top down matching ({}): {:.2f}s, bottom up matches ({}): {:.2f}s'.format(
        t2-t1, n_top_down_matched_nodes, t3-t2, len(bottom_up_matches), t4-t3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help="Which data to process; 'a'-'e' for small examples, "
                        "'exa' for a ~360 line Python file, 'exb' for a ~1600 line Python file with localised changes, "
                        "'exb2' for a 'exb' with changes at the beginning and end")
    args = parser.parse_args()

    A_src, B_src = example_data.get_data(args.data)

    test_fg_match(A_src, B_src)

