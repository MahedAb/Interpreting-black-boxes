"""
The code uses some packages
defined in the pysymbolic
library used in the symbolic
metamodeling approach

For the code of SM and pysymbolic, refer

https://github.com/ahmedmalaa/Symbolic-Metamodeling
"""


from __future__ import absolute_import, division, print_function
from pysymbolic.models.synthetic_datasets import *
from pysymbolic.algorithms.keras_predictive_models import *
import tree_metamodel_exp2
import copy
from pysymbolic.utilities.instancewise_metrics import *
import argparse
import pickle
import time

parser = argparse.ArgumentParser(description="script for the proposed SR-MeijerG approach")
parser.add_argument("--d", default=10, type=int, help="dimension of input")
parser.add_argument("--M", default=12, type=int, help="number of random trees")
parser.add_argument("--s", default=4, type=int, help="number of surviving trees")
parser.add_argument("--max_iter", default=30, type=int, help="number of loop iterations")
parser.add_argument("--l1", default=2, type=int, help="minimum number of middle nodes in a tree")
parser.add_argument("--l2", default=2, type=int, help="maximum number of middle nodes in a tree")
parser.add_argument("--p0", default=0.7, type=float, help="probability of an edge between input and middle nodes")
parser.add_argument("--lam1", default=0.00, type=float, help="regularization term")
parser.add_argument("--p_cross_mut", default=0.7, type=float, help="probability of crossover operation")
parser.add_argument("--num_mut", default=2, type=int, help="number of actions in mutation")
parser.add_argument("--del_prob", default=0.5, type=float, help="probability of deletion in mutation")
parser.add_argument("--seed", default=1234, type=int, help="random seed, for reproducibility")
parser.add_argument("--join_all", default=True, action="store_true",
                    help="if given, join all remaining input features to random middle nodes")
parser.add_argument("--k", default=10, type=int, help="gradient descent updates")
parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
parser.add_argument("--out_dir", default='results', type=str, help="results path")
parser.add_argument("--debug", default=False, action="store_true",
                    help="if given, display debug prints")
args = parser.parse_args()

benchmark_dictionary = {}
datasets = ['XOR', 'nonlinear_additive', 'switch']
benchmark_dictionary['SMGP'] = {}
benchmark_dictionary['SMGP'] = {'XOR': [], 'nonlinear_additive': [], 'switch': []}
num_selected_features = {'XOR': 2,
                         'nonlinear_additive': 4,
                         'switch': 5}


def SMGP_instancewise(x_train, y_train, x_test, model_type):
    predictive_model = get_predictive_model(x_train, y_train, model_type=model_type)

    y_train = predictive_model.predict(x_train)
    s = args.s
    num_operation = int(np.floor(args.M / s) - 1)
    p_cross_mut = args.p_cross_mut
    threshold = 0.0001

    generation_list = []
    for tree_idx in range(args.M):
        n_mid_nodes = np.random.randint(args.l1, args.l2 + 1)
        rt = tree_metamodel_exp2.RandomTree(x_train.shape[1], n_mid_nodes, args.p0, args.join_all, args.debug)
        generation_list.append(rt)
    fittest_tree = [generation_list[0].tree_fitness(x_train, y_train, args.lam1, lam2=0.0), generation_list[0]]

    iteration = 0
    while iteration < args.max_iter and fittest_tree[0] > threshold:
        fitness_score = []
        for item in generation_list:
            item.train_tree(x_train, y_train, args.lr, args.k)
            fitness_score.append(item.tree_fitness(x_train, y_train, args.lam1, lam2=0.0))
        b = np.argsort(fitness_score)
        if fitness_score[b[0]] < fittest_tree[0]:
            fit_tree = copy.deepcopy(generation_list[b[0]])
            fittest_tree = [fitness_score[b[0]], fit_tree]
        survived_list = []
        for i in range(s):
            survived_list.append(generation_list[b[i]])

        operation = np.random.binomial(n=1, p=p_cross_mut, size=[s, num_operation])
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
    scores = []
    for i in range(x_test.shape[0]):
        scores.append(np.abs(np.array(fittest_tree[1].first_order(x_test[i, :]))))
    scores = np.array(scores).reshape((x_test.shape[0], x_train.shape[1]))

    return scores


def get_instancewise_median_ranks(dataset_, num_samples, num_selected_features_, method_):
    model_types_ = {'SMGP': 'sklearn'}

    evaluators_ = {'SMGP': SMGP_instancewise}

    model_type = model_types_[method_]

    eval_method = evaluators_[method_]

    x_train, y_train, x_test, y_test, datatypes_val = create_data(dataset_, n=num_samples)

    eval_scores = eval_method(x_train, y_train, x_test, model_type=model_type)
    eval_ranks = compute_median_rank(eval_scores, k=num_selected_features_, datatype_val=datatypes_val)

    return eval_ranks


# main code starts here

start = time.time()

for dataset in datasets:
    print("---------------")
    print("generating SMGP metamodel for dataset: %s" %dataset)
    benchmark_dictionary['SMGP'][dataset] = get_instancewise_median_ranks(dataset_=dataset, num_samples=1000,
                                                                         num_selected_features_=num_selected_features[
                                                                             dataset], method_='SMGP')

with open(args.out_dir + 'median_smgp.pickle', 'wb') as handle:
    pickle.dump(benchmark_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

stop = time.time()

print("time taken (mins): %f" %((stop-start)/60.0))
