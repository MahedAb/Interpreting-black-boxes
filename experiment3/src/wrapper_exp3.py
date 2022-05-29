"""
The code uses some packages
defined in the pysymbolic
library used in the symbolic
metamodeling approach

For the code of SM and pysymbolic, refer

https://github.com/ahmedmalaa/Symbolic-Metamodeling
"""


import copy
import numpy as np
import argparse
import time
from train_model_exp3 import train_model
from datasets.data_loader_UCI import mixup, data_loader
import tree_metamodel_exp1

# parse command line arguments
parser = argparse.ArgumentParser(description="script for the proposed SR-MeijerG approach")
parser.add_argument("--d", default=10, type=int, help="dimension of input")
parser.add_argument("--M", default=20, type=int, help="number of random trees")
parser.add_argument("--s", default=4, type=int, help="number of surviving trees")
parser.add_argument("--max_iter", default=25, type=int, help="number of loop iterations")
parser.add_argument("--l1", default=1, type=int, help="minimum number of middle nodes in a tree")
parser.add_argument("--l2", default=3, type=int, help="maximum number of middle nodes in a tree")
parser.add_argument("--p0", default=0.6, type=float, help="probability of an edge between input and middle nodes")
parser.add_argument("--k", default=20, type=int, help="gradient descent updates")
parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
parser.add_argument("--lam1", default=0.00001, type=int, help="regularization term")
parser.add_argument("--p_cross_mut", default=0.7, type=int, help="probability of crossover operation")
parser.add_argument("--num_mut", default=3, type=int, help="number of actions in mutation")
parser.add_argument("--del_prob", default=0.5, type=int, help="probability of deletion in mutation")

parser.add_argument("--dataset", default='yacht', type=str, help="dataset name")
parser.add_argument("--bb", default='MLP', type=str, help="black-box model type")

parser.add_argument("--seed", default=1234, type=int, help="random seed, for reproducibility")
parser.add_argument("--join_all", default=True, action="store_true",
                    help="if given, join all remaining input features to random middle nodes")
parser.add_argument("--out_dir", default='results', type=str, help="results path")
parser.add_argument("--debug", default=False, action="store_true",
                    help="if given, display debug prints")
args = parser.parse_args()


def train_trees(x_train, y_train):
    s = args.s
    num_operation = int(np.floor(args.M / s) - 1)
    threshold = 0.0001
    generation_list = []

    for tree_idx in range(args.M):
        n_mid_nodes = np.random.randint(args.l1, args.l2 + 1)
        rt = tree_metamodel_exp1.RandomTree(x_train.shape[1], n_mid_nodes, args.p0, args.join_all, args.debug)
        generation_list.append(rt)
    fittest_tree = [generation_list[0].tree_fitness(x_train, y_train, args.lam1, lam2=0.0), generation_list[0]]
    iteration = 0
    while iteration < args.max_iter and fittest_tree[0] > threshold:
        fitness_score = []
        for item in generation_list:
            item.train_tree(x_train, y_train, args.lr, args.k)
            fitness_score.append(item.tree_fitness(x_train, y_train, args.lam1, lam2=0.0))
        b = np.argsort(fitness_score)
        survived_list = []
        for i in range(s):
            survived_list.append(generation_list[b[i]])
        if fitness_score[b[0]] < fittest_tree[0]:
            fit_tree = copy.deepcopy(generation_list[b[0]])
            fittest_tree = [fitness_score[b[0]], fit_tree]
        operation = np.random.binomial(n=1, p=args.p_cross_mut, size=[s, num_operation])
        new_generation = []
        for k, tree in enumerate(survived_list):
            new_generation.append(tree)
            for j in range(num_operation):
                if operation[k][j]:
                    ind = (k + np.random.randint(1, s)) % s
                    temp_tree = copy.deepcopy(tree.crossover(survived_list[ind]))
                    new_generation.append(temp_tree)
                else:
                    temp_mut = copy.deepcopy(tree.mutate(args.num_mut, args.del_prob))
                    new_generation.append(temp_mut)
        generation_list = copy.deepcopy(new_generation)
        iteration += 1
        #print(fitness_score)
        #print(fittest_tree[0])
    return copy.deepcopy(fittest_tree)

# main code starts from here
dataset = args.dataset
bb = args.bb
print("-----------------------------------------")
print("dataset: %s black_box: %s" %(dataset, bb))
print("-----------------------------------------")

MSE = {'bx': [], 'ourVSbx': [], 'our': []}
R2 = {'bx': [], 'ourVSbx': [], 'our': []}
t1 = time.time()
for i in range(5):
    print()
    print("iteration index: %d" %i)
    X_train, y_train, X_test, y_test = data_loader(dataset_name=dataset)
    model = train_model(X_train, y_train, black_box=bb)
    y_train_MLP = model.predict(X_train)
    y_test_MLP = model.predict(X_test)
    MSE['bx'].append(np.mean((y_test_MLP - y_test) ** 2))

    print("training metamodel....")
    model_tree = train_trees(X_train, y_train_MLP)
    
    print("evaluating metamodel....")
    y_est = model_tree[1].compute_outputs(X_test)
    MSE['ourVSbx'].append(np.mean((y_test_MLP - y_est) ** 2))
    MSE['our'].append(np.mean((y_test - y_est) ** 2))
    R2['bx'].append(1 - (np.mean((y_test - y_test_MLP) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2)))
    R2['our'].append(1 - (np.mean((y_test - y_est) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2)))
    R2['ourVSbx'].append(1 - (np.mean((y_test_MLP - y_est) ** 2) / np.mean((y_test_MLP - np.mean(y_test_MLP)) ** 2)))
for key in MSE:
    MSE_temp = np.array(MSE[key], dtype='float64')
    R2_temp = np.array(R2[key], dtype='float64')
    print(key)
    print('MSE = {} +/- {}'.format(np.mean(MSE_temp), np.std(MSE_temp)))
    print('R2 = {} +/- {}'.format(np.mean(R2_temp), np.std(R2_temp)))
print("M, s, max_iter, p0, k, lr, lam1, l1, l2")
print(args.M, args.s, args.max_iter, args.p0, args.k, args.lr, args.lam1, args.l1, args.l2)
t2 = time.time()
print("time_taken: %f" % (t2 - t1))