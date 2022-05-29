"""
wrapper script
@author: Saumitra Mishra

The code uses some packages
defined in the pysymbolic
library used in the symbolic
metamodeling approach

For the code of SM and pysymbolic, refer

https://github.com/ahmedmalaa/Symbolic-Metamodeling

"""

import numpy as np
import argparse
import tree_metamodel_exp1
import utils
import copy
import sys


# construct the argument parse and parse input arguments
parser = argparse.ArgumentParser(description="script for the proposed SMGP approach")
parser.add_argument("--d", default=2, type=int, help="dimension of input")
parser.add_argument("--M", default=20, type=int, help="number of random trees")
parser.add_argument("--s", default=4, type=int, help="number of surviving trees")
parser.add_argument("--max_iter", default=20, type=int, help="number of loop iterations")
parser.add_argument("--l1", default=1, type=int, help="minimum number of middle nodes in a tree")
parser.add_argument("--l2", default=1, type=int, help="maximum number of middle nodes in a tree")
parser.add_argument("--p0", default=1, type=float, help="probability of an edge between input and middle nodes")
parser.add_argument("--seed", default=1234, type=int, help="random seed, for reproducibility")
parser.add_argument("--join_all", default=True, action="store_true",
                    help="if given, join all remaining input features to random middle nodes")
parser.add_argument("--k", default=10, type=int, help="gradient descent updates")
parser.add_argument("--lr", default=0.2, type=float, help="learning rate")
parser.add_argument("--out_dir", default='results', type=str, help="results path")
parser.add_argument("--debug", default=False, action="store_true",
                    help="if given, display debug prints")

args = parser.parse_args()

print("--------------------------------")
print("number of random trees: %d" % args.M)
print("survived trees: %d" % args.s)
print("min mid nodes: %d" % args.l1)
print("max mid nodes: %d" % args.l2)
print("probability: %.2f" % args.p0)
print("seed: %d" % args.seed)
print("all input connected: %r" % args.join_all)
print("gradient descent updates: %d" % args.k)
print("learning rate: %.3f" % args.lr)
print("results dir: %s" % args.out_dir)
print("maximum loop iterations: %d" % args.max_iter)
print("display debug prints: %r" % args.debug)


# reproducibility
np.random.seed(args.seed)

# 1. read input features and labels
X_data, Y_data = utils.create_data_splits()
if args.debug:
    print("[training data] dimensions: %d, samples: %d" % (X_data['train'].shape[1], X_data['train'].shape[0]))
    print("[test data] dimensions: %d, samples: %d" % (X_data['test'].shape[1], X_data['test'].shape[0]))
    print("--------------------------------")

Num_operation = int(np.floor(args.M / args.s) - 1)  # Number of required operations on each survived tree
p_cross_mut = 0.8  # The probability of crossover operation (1-p_cross_mut) is probability of mutation operation
threshold = 0.00001

generation_list = []  # list of random trees

for true_func in Y_data:  # gives key

    print("--------------------------------")
    print("modeling the function: %s" % true_func)

    # 2. initialise M random trees each with middle nodes in [l1, l2]
    for tree_idx in range(args.M):
        print("====random tree index: %d====" % tree_idx)

        n_mid_nodes = np.random.randint(args.l1, args.l2 + 1)
        print("number of middle layer nodes: %d" % n_mid_nodes)

        # initialise
        print("initialisation....")
        rt = tree_metamodel_exp1.RandomTree(args.d, n_mid_nodes, args.p0, args.join_all, args.debug)

        # save the tree
        generation_list.append(rt)

    # 3. fittest tree calculation
    fittest_tree = [generation_list[0].tree_fitness(X_data['train'], Y_data[true_func]['train'],
                                                    lam1=0.0, lam2=0.0), generation_list[0]]


    # 4. train trees and identify the fittest tree
    print("++++++++++++++++++")
    print("training trees....")
    iteration = 0
    while iteration < args.max_iter and fittest_tree[0] > threshold:
        print("iteration index: %d" % iteration)
        fitness_score = []
        print("score: %.3f" %fittest_tree[0])
        for item in generation_list:
            item.train_tree(X_data['train'], Y_data[true_func]['train'], args.lr, args.k)
            fitness_score.append(item.tree_fitness(X_data['train'], Y_data[true_func]['train'], lam1=0.0, lam2=0.0))

        b = np.argsort(fitness_score)
        if fitness_score[b[0]] < fittest_tree[0]:
            fit_tree = copy.deepcopy(generation_list[b[0]])
            fittest_tree = [fitness_score[b[0]], fit_tree]

        survived_list = []
        for i in range(args.s):
            survived_list.append(generation_list[b[i]])

        operation = np.random.binomial(n=1, p=p_cross_mut, size=[args.s, Num_operation])
        new_generation = []
        for k, tree in enumerate(survived_list):
            new_generation.append(tree)
            for j in range(Num_operation):
                if operation[k][j]:
                    ind = (k + np.random.randint(1, args.s)) % args.s
                    temp_tree = copy.deepcopy(tree.crossover(survived_list[ind]))
                    new_generation.append(temp_tree)
                else:
                    temp_mut = copy.deepcopy(tree.mutate(1, 0))
                    new_generation.append(temp_mut)
        generation_list = copy.deepcopy(new_generation)
        iteration += 1
        sys.stdout.flush()

    # 5. R2 and MSE computation on training data
    R2_score, mean_squared_error = utils.compute_performance(X_data['train'],
                                                             Y_data[true_func]['train'],
                                                             fittest_tree[1])
    print("++++++++++++++++++")
    print("performance computation....")
    print("R2 score[train]: %f" % R2_score)
    print("mean square error[train]: %f" % mean_squared_error)

    # R2 and MSE computation on the test data
    test_batch_size = 100
    agg_test_r2 = list()
    agg_test_mse = list()
    for i in range(int(X_data['test'].shape[0]/test_batch_size)):
        start = i*test_batch_size
        stop = test_batch_size*(i+1)
        R2_score, mean_squared_error = utils.compute_performance(X_data['test'][start:stop, :],
                                                                 Y_data[true_func]['test'][start:stop],
                                                                 fittest_tree[1])

        agg_test_r2.append(R2_score)
        agg_test_mse.append(mean_squared_error)
    print("R2 score [test] mean: %f  std dev: %f " % (np.mean(np.array(agg_test_r2)), np.std(np.array(agg_test_r2))))
    print("MSE score [test] mean: %f  std dev: %f " % (np.mean(np.array(agg_test_mse)), np.std(np.array(agg_test_mse))))
    print("++++++++++++++++++")
    print("metamodel expression: ", end='')
    print(fittest_tree[1].Kolmogorov_expression())
    
    sys.stdout.flush()