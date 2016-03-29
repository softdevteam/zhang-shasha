import sys
import pygit2



import time, collections, sys, argparse

from zss import simple_tree, compare, zs_memo

import parser, ast, _ast

import type_pruning

from ast_to_simple_tree import ASTConverter
from source_text import SourceText, Marker, longest_common_prefix, longest_common_suffix
import tree_flattening


def compute_shape_fingerprints(A, B):
    fingerprints = {}
    A_nodes_by_index = {}
    B_nodes_by_index = {}
    A.update_fingerprint_index(fingerprints, A_nodes_by_index)
    B.update_fingerprint_index(fingerprints, B_nodes_by_index)
    return fingerprints, A_nodes_by_index, B_nodes_by_index


def tree_diff(A_src, B_src):
    conv = ASTConverter()


    try:
        A = conv.parse(A_src)
        B = conv.parse(B_src)
    except SyntaxError:
        return None, None


    A_opt = A
    B_opt = B


    comparison_permitted_by_label = None


    base_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)


    unique_node_matches = None
    potential_match_fingerprints = None

    A_nodes_to_collapse = set()
    B_nodes_to_collapse = set()
    for A_fg_index, A_nodes in A_nodes_by_index.items():
        B_nodes = B_nodes_by_index.get(A_fg_index)
        if B_nodes is not None:
            if len(A_nodes) == len(B_nodes):
                A_nodes_to_collapse = A_nodes_to_collapse.union(A_nodes)
                B_nodes_to_collapse = B_nodes_to_collapse.union(B_nodes)
    A_opt = A_opt.compact(A_nodes_to_collapse)
    B_opt = B_opt.compact(B_nodes_to_collapse)



    opt_fingerprints, A_nodes_by_index, B_nodes_by_index = compute_shape_fingerprints(A_opt, B_opt)


    t1 = time.time()
    d, node_matches = zs_memo.simple_distance(A_opt, B_opt, len(opt_fingerprints), simple_tree.Node.get_children, simple_tree.Node.get_label,
                                comparison_filter=comparison_permitted_by_label,
                                unique_match_constraints=unique_node_matches,
                                potential_match_fingerprints=potential_match_fingerprints)
    compare.check_match_list(node_matches)
    t2 = time.time()

    return d, t2-t1



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




def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '.'
    repo = pygit2.Repository(path)
    master_branch = repo.lookup_branch('master')
    master_commit = repo[master_branch.resolve().target]
    master_tree = master_commit.tree
    walk_commit(repo, master_branch.resolve().target)


if __name__ == '__main__':
    main()