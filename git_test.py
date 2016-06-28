import argparse, os, parser, subprocess, sys, tempfile, time, json

import pygit2

from ast_to_simple_tree import ASTConverter
import python_ast_structure
from zss import compare, zs_memo, tree_match, edit_script, fg_match
from zss.source_text import SourceText, longest_common_prefix, longest_common_suffix



def tree_diff_zs(A_src, A, B_src, B, verbose):
    if verbose:
        print 'SOURCE CODE STATS: |A_src|={0}, |B_src|={1}, |A_src.lines|={2}, |B_src.lines|={3}, ' \
              '|common prefix lines|={4}, |common suffix lines|={5}'.format(
            len(A_src), len(B_src), len(A_src.lines), len(B_src.lines), longest_common_prefix(A_src.lines, B_src.lines),
            longest_common_suffix(A_src.lines, B_src.lines))

    d, node_matches, dt = tree_match.tree_match(A, B, verbose=verbose, distance_fn=zs_memo.simple_distance)

    return d, node_matches, dt



def test_fg_match(A_src, A, B_src, B, verbose):
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

    node_matches = top_down_matches + bottom_up_matches


    return -1, node_matches, t4-t1



def tree_diff_test(A_src, B_src, verbose=False, use_fg=False):
    conv = ASTConverter()

    try:
        A = conv.parse(A_src)
        B = conv.parse(B_src)
    except SyntaxError:
        return None, None

    if verbose:
        print 'SOURCE CODE STATS: |A_src|={0}, |B_src|={1}, |A_src.lines|={2}, |B_src.lines|={3}, ' \
              '|common prefix lines|={4}, |common suffix lines|={5}'.format(
            len(A_src), len(B_src), len(A_src.lines), len(B_src.lines), longest_common_prefix(A_src.lines, B_src.lines),
            longest_common_suffix(A_src.lines, B_src.lines))


    if use_fg:
        d, node_matches, dt = test_fg_match(A_src, A, B_src, B, verbose)
    else:
        d, node_matches, dt = tree_diff_zs(A_src, A, B_src, B, verbose)


    compare.check_match_list(node_matches)

    diffs = edit_script.edit_script(A, B, node_matches)


    X = A.clone()
    A_ids = {a.merge_id for a in A.iter()}
    X_ids = {x.merge_id for x in X.iter()}
    assert A_ids == X_ids
    merge_id_to_node = {}
    for node in X.iter():
        merge_id_to_node[node.merge_id] = node
    for diff in diffs:
        diff.apply(merge_id_to_node)

    assert X == B


    return d, dt





def show_tree(prefix, repo, tree):
    for entry in tree:
        data = repo[entry.id]
        print '{0}/{1}: {2}  {3}'.format(prefix, entry.name, entry, data)
        if isinstance(data, pygit2.Tree):
            show_tree(prefix + '/' + entry.name, repo, data)



def walk_commit(repo, commit_id, use_fg):
    num_diffs = 0
    min_dt = None
    max_dt = None
    sum_dt = None
    bad_commits_paths_and_times = []
    for commit in repo.walk(commit_id, pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_REVERSE):
        for parent in commit.parents:
            # print 'DIFF FROM {0} to {1}'.format(parent, commit)
            diff = repo.diff(commit.tree, parent.tree)
            for patch in diff:
                old_file = patch.delta.old_file
                new_file = patch.delta.new_file
                old_blob = repo.get(old_file.id)
                new_blob = repo.get(new_file.id)
                if old_file.path.endswith('.py') and new_file.path.endswith('.py'):
                    # print '{}({}) became {}({})'.format(old_file.path, old_file.id, new_file.path, new_file.id)
                    if old_blob is not None and new_blob is not None:
                        d, dt = tree_diff_test(SourceText(old_blob.data), SourceText(new_blob.data), use_fg=use_fg)
                        if d is not None:
                            min_dt = min(dt, min_dt) if min_dt is not None else dt
                            max_dt = max(dt, max_dt) if max_dt is not None else dt
                            sum_dt = sum_dt + dt if sum_dt is not None else dt
                            num_diffs += 1
                            if dt > 1.0:
                                bad_commits_paths_and_times.append((commit.id, parent.id, old_file.path, dt))
                            sys.stdout.write('.')
                            if num_diffs % 100 == 0:
                                sys.stdout.write('({})'.format(num_diffs))
                            sys.stdout.flush()
    print ''
    print '{0} diffs took {1}; min={2}, max={3}, avg={4}'.format(num_diffs, sum_dt, min_dt, max_dt, sum_dt / num_diffs)
    for commit, parent, path, dt in bad_commits_paths_and_times:
        print 'Commit {0} from {1}: file {2} took {3}'.format(commit, parent, path, dt)




def walk_master(repo_path, use_fg):
    repo = pygit2.Repository(repo_path)
    master_branch = repo.lookup_branch('master')
    master_commit = repo[master_branch.resolve().target]
    walk_commit(repo, master_branch.resolve().target, use_fg=use_fg)


def compare_file(repo_path, path, commit_id, parent_commit_id, use_diff, use_fg):
    repo = pygit2.Repository(repo_path)
    if commit_id is None:
        commit = repo.revparse_single('HEAD')
    else:
        commit = repo.get(commit_id)

    if parent_commit_id is None:
        parent = commit.parents[0]
    else:
        parent = repo.get(parent_commit_id)
    if commit is not None and parent is not None:
        new_file = commit.tree[path]
        old_file = parent.tree[path]
        old_blob = repo.get(old_file.id)
        new_blob = repo.get(new_file.id)
        if use_diff:
            a_fd, a_name = tempfile.mkstemp()
            b_fd, b_name = tempfile.mkstemp()
            a_f = os.fdopen(a_fd, 'w')
            b_f = os.fdopen(b_fd, 'w')
            a_f.write(old_blob.data)
            b_f.write(new_blob.data)
            a_f.close()
            b_f.close()
            t1 = time.time()
            proc = subprocess.Popen(['diff', a_name, b_name], stdout=subprocess.PIPE)
            proc.communicate()
            os.remove(a_name)
            os.remove(b_name)
            t2 = time.time()
            dt = t2 - t1
        else:
            d, dt = tree_diff_test(SourceText(old_blob.data), SourceText(new_blob.data), verbose=True, use_fg=use_fg)
        print 'Took {0:.3f}s'.format(dt)


def export_file(repo_path, path, commit_id, out_path):
    repo = pygit2.Repository(repo_path)
    if commit_id is None:
        commit = repo.revparse_single('HEAD')
    else:
        commit = repo.get(commit_id)

    if commit is not None:
        repo_file = commit.tree[path]
        repo_blob = repo.get(repo_file.id)
        src = SourceText(repo_blob.data)
        conv = ASTConverter()
        tree = conv.parse(src)
        js = tree.as_json_tree(python_ast_structure.ast_class_name_to_id_map)
        js_text = json.dumps(js, indent=2, sort_keys=True)
        if out_path is None:
            sys.stdout.write(js_text)
            sys.stdout.flush()
        else:
            open(out_path, 'w').write(js_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help="Command: [treediff/filediff/export]")
    parser.add_argument('repo_path', type=str, help="Git repository path")
    parser.add_argument('--file', type=str, default=None, help="File path")
    parser.add_argument('--commit', type=str, default=None, help="Commit ID")
    parser.add_argument('--parent', type=str, default=None, help="Parent commit ID")
    parser.add_argument('--diff', action='store_true', help="Just use diff")
    parser.add_argument('--use_fg', action='store_true', help="Use fingerprint matching")
    parser.add_argument('--out', type=str, default=None, help="Output path, None for STDOUT")
    args = parser.parse_args()

    if args.command == 'treediff':
        walk_master(args.repo_path, args.use_fg)
    elif args.command == 'filediff':
        compare_file(args.repo_path, args.file, args.commit, args.parent, args.diff, args.use_fg)
    elif args.command == 'export':
        export_file(args.repo_path, args.file, args.commit, args.out)
