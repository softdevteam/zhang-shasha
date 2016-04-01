import sys, os
import pygit2

import tempfile

import subprocess



import time, collections, sys, argparse

from zss import simple_tree, compare, zs_memo, tree_match, edit_script

import parser, ast, _ast

import type_pruning

from ast_to_simple_tree import ASTConverter
from source_text import SourceText, Marker, longest_common_prefix, longest_common_suffix
import tree_flattening


def tree_diff(A_src, B_src, verbose=False):
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

    d, node_matches, dt = tree_match.tree_match(A, B, verbose=verbose, distance_fn=zs_memo.simple_distance)
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



def walk_commit(repo, commit_id):
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
                        d, dt = tree_diff(SourceText(old_blob.data), SourceText(new_blob.data))
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




def walk_master(repo_path):
    repo = pygit2.Repository(repo_path)
    master_branch = repo.lookup_branch('master')
    master_commit = repo[master_branch.resolve().target]
    walk_commit(repo, master_branch.resolve().target)


def compare_file(repo_path, path, commit_id, parent_commit_id, use_diff):
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
            d, dt = tree_diff(SourceText(old_blob.data), SourceText(new_blob.data), verbose=True)
        print 'Took {0:.3f}s'.format(dt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_path', type=str, help="Git repository path")
    parser.add_argument('--file', type=str, default=None, help="File path")
    parser.add_argument('--commit', type=str, default=None, help="Commit ID")
    parser.add_argument('--parent', type=str, default=None, help="Parent commit ID")
    parser.add_argument('--diff', action='store_true', help="Just use diff")
    args = parser.parse_args()

    if args.file is not None:
        compare_file(args.repo_path, args.file, args.commit, args.parent, args.diff)
    else:
        walk_master(args.repo_path)