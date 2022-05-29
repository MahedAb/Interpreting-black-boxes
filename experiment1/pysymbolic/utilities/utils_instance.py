import numpy as np


def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.

    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank = (-permutated_weights).argsort().argsort() + 1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)

    return np.array(ranks)


def compute_median_rank(scores, k, datatype_val=None):
    ranks = create_rank(scores, k)
    if datatype_val is None:
        median_ranks = np.median(ranks[:, :k], axis=1)
    else:
        datatype_val = datatype_val[:len(scores)]
        median_ranks1 = np.median(ranks[datatype_val == 'orange_skin', :][:, np.array([0, 1, 2, 3, 9])],
                                  axis=1)
        median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive', :][:, np.array([4, 5, 6, 7, 9])],
                                  axis=1)
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    return median_ranks

def Our_instancewise(x_train, y_train, x_test, model_type):
    predictive_model = get_predictive_model(x_train, y_train, model_type=model_type)

    y_train = predictive_model.predict(x_train)
    M = 2
    s = 2
    Num_operation = int(np.floor(M / s) - 1)
    p_cross_mut = 0.8
    threshold = 0.0001
    generation_list = []
    p0 = 0.5
    Join_all = 1
    max_iter = 1
    for tree_idx in range(M):
        n_mid_nodes = 2  # np.random.randint(1, 1 + 1)
        rt = tree_S_S.RandomTree(x_train.shape[1], n_mid_nodes,
                                 p0, Join_all, False)
        generation_list.append(rt)
    fittest_tree = [generation_list[0].tree_fitness(x_train, y_train,
                                                    lam1=0.0, lam2=0.0), generation_list[0]]
    iteration = 0
    while iteration < max_iter and fittest_tree[0] > threshold:
        fitness_score = []
        for item in generation_list:
            item.train_tree(x_train, y_train, 1e-3, 2)
            fitness_score.append(item.tree_fitness(x_train, y_train, lam1=0.0, lam2=0.0))
        b = np.argsort(fitness_score)
        survived_list = []
        for i in range(s):
            survived_list.append(generation_list[b[i]])

        operation = np.random.binomial(n=1, p=p_cross_mut, size=[s, Num_operation])
        new_generation = []
        for k, tree in enumerate(survived_list):
            new_generation.append(tree)
            for j in range(Num_operation):
                if operation[k][j]:
                    ind = (k + np.random.randint(1, s)) % s
                    temp_tree = copy.deepcopy(tree.crossover(survived_list[ind]))
                    new_generation.append(temp_tree)
                else:
                    temp_mut = copy.deepcopy(tree.mutate(1, 0))
                    new_generation.append(temp_mut)
        generation_list = copy.deepcopy(new_generation)
        iteration += 1
    scores = []
    for i in range(x_test.shape[0]):
        scores.append(np.abs(np.array(fittest_tree[1].first_order(x_test[i, :]))))
    scores = np.array(scores).reshape((x_test.shape[0], x_train.shape[1]))

    return scores


def get_instancewise_median_ranks(dataset_, num_samples, num_selected_features, method_):
    model_types_ = {'SR': "keras",
                    'LIME': "modified_keras",
                    'SHAP': "sklearn",
                    'DeepLIFT': "keras",
                    'SM': "sklearn",
                    'SP': "sklearn",
                    'Our': 'sklearn'}

    evaluators_ = {'Our': Our_instancewise}

    model_type = model_types_[method_]

    eval_method = evaluators_[method_]

    x_train, y_train, x_test, y_test, datatypes_val = create_data(dataset_, n=num_samples)

    eval_scores = eval_method(x_train, y_train, x_test, model_type=model_type)
    eval_ranks = compute_median_rank(eval_scores, k=num_selected_features, datatype_val=datatypes_val)

    return eval_ranks