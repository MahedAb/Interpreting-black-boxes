from sympy import *
from pysymbolic.models.special_functions import MeijerG
import copy
import numpy as np
from sympy.abc import x
from sympy.functions import sin, exp, log

a0, a1, a2, a3 = symbols('a0 a1 a2 a3')


class RandomTree:
    def __init__(self, d, n_mid, prob, join_rem_features, debug, verbosity=True):
        """
        initialises class variables and edge parameters
        """

        self.dim = d
        self.n_mid_nodes = n_mid
        self.print = debug
        self.verbosity = verbosity  # True if the optimization process should be detailed
        self.classes_list = ["exp", "poly", "sin", "fraction"]
        self.classes_dict = self.read_classes()
        # params for edges between the mid and out layers [(node_idx, params, order), .... ]
        self.params_mid_out = self.init_params_mid_out()
        # information about connections between the input and middle layers [[node_idx, [binary_list]], .... ]
        self.mid_inp_edges = self.create_mid_inp_edges(prob)

        if self.print:
            print("init params for mid-top edges: ", end='')
            print(self.params_mid_out)
            print("edges for mid-input: ", end='')
            print(self.mid_inp_edges)

        # if join_rem_features => connect all remaining input features randomly to middle nodes
        if join_rem_features:
            self.update_mid_inp_edges()
            if self.print:
                print("edges for mid-input (after connecting all features): ", end='')
                print(self.mid_inp_edges)

        # for each middle node populate the parameters for the mid-inp edges
        self.params_mid_inp = self.init_params_mid_inp()
        if self.print:
            print("init params for inp-mid edges: ", end='')
            print(self.params_mid_inp)

        self.e = self.number_edges()

    def read_classes(self):
        """
        :return:the fixed Meijer-G hyperparameters
        """
        my_exp = a0 * exp(-a1 * x)
        my_pol = a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0
        my_sin = a0 * sin(a1 * x + a2)
        my_log = a0 * log(a1 * x + a2)
        my_fraction = a0 * x / (a1 * x ** 2 + a2 * x + a3)
        return {"exp": (2, my_exp, diff(my_exp, x), diff(my_exp, a0), diff(my_exp, a1)),  # [a,b] a e^-bx   [a*exp(-bx)
                "poly": (4, my_pol, diff(my_pol, x), diff(my_pol, a0), diff(my_pol, a1), diff(my_pol, a2),
                         diff(my_pol, a3)),  # [a,b,c,d] ax^3+bx^2+cx+d
                "sin": (3, my_sin, diff(my_sin, x), diff(my_sin, a0), diff(my_sin, a1),
                        diff(my_sin, a2)),  # [a,b,c] a sin (bx+c)[0.0, 0.0, 0.0]
                "log": (3, my_log, diff(my_log, x), diff(my_log, a0), diff(my_log, a1),
                        diff(my_log, a2)),  # includes log and many more, need to take real value though
                "fraction": (4, my_fraction, diff(my_fraction, x), diff(my_fraction, a0), diff(my_fraction, a1),
                             diff(my_fraction, a2),
                             diff(my_fraction, a3))}  # Parent of Bessel functions

    def init_params_mid_out(self):
        """
        function to initialise the parameters for
        edges between the mid and the out layers
        """
        params = []
        for mid_node in range(self.n_mid_nodes):
            # randomly select a class of function and assign initial values
            type_func = self.classes_list[np.random.randint(0, len(self.classes_list))]
            params.append([type_func, self.classes_dict[type_func][0], np.random.rand(self.classes_dict[type_func][0])])
            if self.print:
                print("[mid-out] params: ", end='')
                print(type_func)
        return params

    def create_mid_inp_edges(self, inp_prob):
        """
        function to create edges between
        input and middle nodes
        """
        mid_inp_edges = []
        for mid_node in range(self.n_mid_nodes):
            edges = np.random.binomial(n=1, p=inp_prob, size=self.dim)  # n=1 => bernoulli distribution
            mid_inp_edges.append([mid_node, edges.tolist()])
        return mid_inp_edges

    def update_mid_inp_edges(self):
        """
        function connects all remaining
        input features to uniformly randomly
        selected middle nodes and updates the
        binary connection lists for the selected
        middle nodes
        """
        # find unconnected input features
        features = self.features_not_connected()
        if self.print:
            print("indices of unconnected input features: ", end='')
            print(features)

        # update the binary connection list
        for feature in features:
            mid_node_idx = np.random.randint(0, self.n_mid_nodes)  # samples uniformly
            current_edges = self.mid_inp_edges[mid_node_idx][1]
            current_edges[feature] = 1
            self.mid_inp_edges[mid_node_idx][1] = current_edges

    def features_not_connected(self):
        """
        function to compute the indices
        of unconnected input features
        """
        sum_is_edge = np.zeros(self.dim)
        for i in range(len(self.mid_inp_edges)):
            sum_is_edge += np.asarray(self.mid_inp_edges[i][1])
        return np.where(sum_is_edge == 0)[0].tolist()

    def init_params_mid_inp(self):
        """
        function to initialise the
        parameters of the connections
        between the input and middle nodes
        """
        params = []
        for mid_node in range(self.n_mid_nodes):
            params_mid_inp_pair = []
            for feat_idx, edge in enumerate(self.mid_inp_edges[mid_node][1]):
                if edge:  # edge exists
                    # randomly select Meijer-G hyper-parameters for the edge
                    type_func = self.classes_list[np.random.randint(0, len(self.classes_list))]
                    if self.print:
                        print("[inp-mid] params: ", end='')
                        print(type_func)
                    params_mid_inp_pair.append([mid_node, feat_idx, type_func, self.classes_dict[type_func][0],
                                                np.random.rand(self.classes_dict[type_func][0])])
            params.append(params_mid_inp_pair)
        return params

    def number_edges(self):
        e = 0
        for edge in range(self.n_mid_nodes):
            e = e + sum(self.mid_inp_edges[edge][1]) + 1
        return e

    def train_tree(self, x_batch, y_batch, lr=0.01, k=10):
        for r in range(k):
            # Computing mid nodes value
            mid_nodes = []  # List of 1-d arrays of shape input.shape[0]
            for mid_inp_edges in self.params_mid_inp:
                sum_mid_inp_edges = np.zeros(x_batch.shape[0])
                for edge in mid_inp_edges:
                    if edge[3] == 2:
                        expr = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1])])
                    elif edge[3] == 3:
                        expr = self.classes_dict[edge[2]][1].subs(
                            [(a0, edge[4][0]), (a1, edge[4][1]), (a2, edge[4][2])])
                    elif edge[3] == 4:
                        expr = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]), (a2, edge[4][2]),
                                                                   (a3, edge[4][3])])
                    f = lambdify(x, expr, "numpy")
                    sum_mid_inp_edges += f(x_batch[:, edge[1]])
                mid_nodes.append(sum_mid_inp_edges)

            # Computing current function value
            func_curr = np.zeros(x_batch.shape[0])
            for ind, edge in enumerate(self.params_mid_out):
                if edge[1] == 2:
                    expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1])])
                elif edge[1] == 3:
                    expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2])])
                elif edge[1] == 4:
                    expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2]),
                                                               (a3, edge[2][3])])
                f = lambdify(x, expr, "numpy")
                func_curr += f(mid_nodes[ind])

            # Computing gradients of mid_inp edges
            grads_in = []
            temp_params = []
            for ind, mid_node_to_inp in enumerate(self.params_mid_inp):
                out_edge = self.params_mid_out[ind]
                if out_edge[1] == 2:
                    expr = self.classes_dict[out_edge[0]][2].subs([(a0, out_edge[2][0]), (a1, out_edge[2][1])])
                elif out_edge[1] == 3:
                    expr = self.classes_dict[out_edge[0]][2].subs([(a0, out_edge[2][0]), (a1, out_edge[2][1]),
                                                                   (a2, out_edge[2][2])])
                elif out_edge[1] == 4:
                    expr = self.classes_dict[out_edge[0]][2].subs([(a0, out_edge[2][0]), (a1, out_edge[2][1]),
                                                                   (a2, out_edge[2][2]), (a3, out_edge[2][3])])
                f = lambdify(x, expr, "numpy")
                f_der = f(mid_nodes[ind])
                temp_params.append([])
                grads_in.append([])
                for idx, edge in enumerate(mid_node_to_inp):
                    temp_params[ind].append(edge)
                    if edge[3] == 2:
                        for i in range(edge[3]):
                            expr = self.classes_dict[edge[2]][i + 3].subs([(a0, edge[4][0]), (a1, edge[4][1])])
                            f = lambdify(x, expr, "numpy")
                            grad_edge = f(x_batch[:, edge[1]]) * f_der
                            grad_edge1 = np.mean(2 * grad_edge * (func_curr - y_batch))
                            grads_in[ind].append(grad_edge1)
                            grad_edge1 = np.clip(grad_edge1, a_min=-50, a_max=50)
                            temp_params[ind][idx][4][i] = edge[4][i] - lr * grad_edge1
                    elif edge[3] == 3:
                        for i in range(edge[3]):
                            expr = self.classes_dict[edge[2]][i + 3].subs([(a0, edge[4][0]),
                                                                           (a1, edge[4][1]), (a2, edge[4][2])])
                            f = lambdify(x, expr, "numpy")
                            grad_edge = f(x_batch[:, edge[1]]) * f_der
                            grad_edge1 = np.mean(2 * grad_edge * (func_curr - y_batch))
                            grads_in[ind].append(grad_edge1)
                            grad_edge1 = np.clip(grad_edge1, a_min=-100, a_max=100)
                            temp_params[ind][idx][4][i] = edge[4][i] - lr * grad_edge1
                    elif edge[3] == 4:
                        for i in range(edge[3]):
                            expr = self.classes_dict[edge[2]][i + 3].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                           (a2, edge[4][2]), (a3, edge[4][3])])
                            f = lambdify(x, expr, "numpy")
                            grad_edge = f(x_batch[:, edge[1]]) * f_der
                            grad_edge1 = np.mean(2 * grad_edge * (func_curr - y_batch))
                            grads_in[ind].append(grad_edge1)
                            grad_edge1 = np.clip(grad_edge1, a_min=-100, a_max=100)
                            temp_params[ind][idx][4][i] = edge[4][i] - lr * grad_edge1
            self.params_mid_inp = copy.deepcopy(temp_params)

            # Computing gradient of mid_out edges
            grads_out = []
            temp_params = []
            for ind, edge in enumerate(self.params_mid_out):
                grads_out.append([])
                temp_params.append(edge)
                if edge[1] == 2:
                    for i in range(edge[1]):
                        expr = self.classes_dict[edge[0]][i + 3].subs([(a0, edge[2][0]), (a1, edge[2][1])])
                        f = lambdify(x, expr, "numpy")
                        grad = np.mean(2 * f(mid_nodes[ind]) * (func_curr - y_batch))
                        grads_out[ind].append(grad)
                        grad = np.clip(grad, a_min=-100, a_max=100)
                        temp_params[ind][2][i] = edge[2][i] - lr * grad
                elif edge[1] == 3:
                    for i in range(edge[1]):
                        expr = self.classes_dict[edge[0]][i + 3].subs([(a0, edge[2][0]),
                                                                       (a1, edge[2][1]), (a2, edge[2][2])])
                        f = lambdify(x, expr, "numpy")
                        grad = np.mean(2 * f(mid_nodes[ind]) * (func_curr - y_batch))
                        grads_out[ind].append(grad)
                        grad = np.clip(grad, a_min=-100, a_max=100)
                        temp_params[ind][2][i] = edge[2][i] - lr * grad
                elif edge[1] == 4:
                    for i in range(edge[1]):
                        expr = self.classes_dict[edge[0]][i + 3].subs([(a0, edge[2][0]), (a1, edge[2][1]),
                                                                       (a2, edge[2][2]), (a3, edge[2][3])])
                        f = lambdify(x, expr, "numpy")
                        grad = np.mean(2 * f(mid_nodes[ind]) * (func_curr - y_batch))
                        grads_out[ind].append(grad)
                        grad = np.clip(grad, a_min=-100, a_max=100)
                        temp_params[ind][2][i] = edge[2][i] - lr * grad

            self.params_mid_out = copy.deepcopy(temp_params)

    def compute_outputs(self, x_batch):
        """
        function to compute outputs
        for a random tree
        """
        # 1. compute middle nodes
        if self.print:
            print("evaluating mid-inp edges...")

        mid_nodes = []  # List of 1-d arrays of shape input.shape[0]
        for mid_inp_edges in self.params_mid_inp:
            sum_mid_inp_edges = np.zeros(x_batch.shape[0])
            for edge in mid_inp_edges:
                if edge[3] == 2:
                    expr = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1])])
                elif edge[3] == 3:
                    expr = self.classes_dict[edge[2]][1].subs(
                        [(a0, edge[4][0]), (a1, edge[4][1]), (a2, edge[4][2])])
                elif edge[3] == 4:
                    expr = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]), (a2, edge[4][2]),
                                                               (a3, edge[4][3])])
                f = lambdify(x, expr, "numpy")
                sum_mid_inp_edges += f(x_batch[:, edge[1]])
            mid_nodes.append(sum_mid_inp_edges)

        # 2. compute the output node
        if self.print:
            print("evaluating mid-out edges...")
        out_node = np.zeros(x_batch.shape[0])
        for ind, edge in enumerate(self.params_mid_out):
            if edge[1] == 2:
                expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1])])
            elif edge[1] == 3:
                expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2])])
            elif edge[1] == 4:
                expr = self.classes_dict[edge[0]][1].subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2]),
                                                           (a3, edge[2][3])])
            f = lambdify(x, expr, "numpy")
            out_node += f(mid_nodes[ind])
        return out_node

    def tree_fitness(self, x_input, y_true, lam1=0.1, lam2=0.1):
        """
        This function compute fitness of a given tree
        Args:
            func_inp:
            self: A tree
            x_input: Random sample of inputs
            y_true: true y
            func_inp: True function (black box)
            lam1: hyper-parameter for contribution of penalty on complexity of zeros and poles (m,n,p,q)
            lam2: Penalty for number of edges (encourage sparse trees)
        Returns: value of fitness
        """
        y_pred = self.compute_outputs(x_input).reshape((-1, 1))
        loss = np.mean((y_true.reshape((-1, 1)) - y_pred) ** 2)
        return loss + lam1 * self.e

    def crossover(self, t):
        """
        This function take two trees and produce a new one by crossover operation
        Args:
            self: The base tree
            t: The other tree, from which a single node and edges will be copied
        Returns: The new produced tree
        """

        ind1 = np.random.randint(self.n_mid_nodes)
        ind2 = np.random.randint(t.n_mid_nodes)
        t_new = copy.deepcopy(self)
        t_new.params_mid_out[ind1] = copy.deepcopy(t.params_mid_out[ind2])
        t_new.e -= (sum(t_new.mid_inp_edges[ind1][1]) - sum(t.mid_inp_edges[ind2][1]))
        t_new.mid_inp_edges[ind1][1] = list(t.mid_inp_edges[ind2][1])
        t_new.params_mid_inp[ind1] = copy.deepcopy(t.params_mid_inp[ind2])
        for edge in t_new.params_mid_inp[ind1]:
            edge[0] = ind1

        return t_new

    def mutate(self, num_mutations=1, p_del=0.2, p_ins=0):
        """
        This function creates a mutation
        Args:
            self: input tree
            num_mutations: how many mutations to make
            p_del: probability of deleting an edge as one of the mutations
            p_ins: probability of inserting an edge as one of the mutations (probably keep it 0)
        Returns: mutated version of the input tree
        """
        t_new = copy.deepcopy(self)
        edge_list = np.random.randint(t_new.e, size=num_mutations)
        del_edge = np.random.binomial(n=1, p=p_del, size=num_mutations)
        ins_edge = np.random.binomial(n=1, p=p_ins, size=num_mutations)
        for j in range(num_mutations):

            if del_edge[j] and (t_new.e - t_new.n_mid_nodes) > 1:
                flat_list = [item for sublist in t_new.params_mid_inp for item in sublist]
                ind_mutate = np.random.randint(len(flat_list))
                node = flat_list[ind_mutate][0]
                ind_rand = np.random.randint(len(t_new.params_mid_inp[node]))
                feature = t_new.params_mid_inp[node][ind_rand][1]
                del t_new.params_mid_inp[node][ind_rand]
                t_new.mid_inp_edges[node][1][feature] = 0
                t_new.e -= 1
                if sum(t_new.mid_inp_edges[node][1]) == 0:
                    for ii in range(node + 1, t_new.n_mid_nodes):
                        t_new.mid_inp_edges[ii][0] -= 1
                        for edge in t_new.params_mid_inp[ii]:
                            edge[0] -= 1
                    
                    del t_new.params_mid_out[node]
                    del t_new.mid_inp_edges[node]
                    t_new.n_mid_nodes -= 1
                    t_new.e -= 1
                    del t_new.params_mid_inp[node]
                edge_list = np.random.randint(t_new.e, size=num_mutations)
            elif ins_edge[j]:
                pass
            else:
                if edge_list[j] < t_new.n_mid_nodes:
                    type_func = self.params_mid_out[edge_list[j]][0]
                    for num, item in enumerate(self.classes_list):
                        if type_func == item:
                            hyper_old = num
                    hyper_new = (hyper_old + np.random.randint(1, 4)) % 4
                    type_new = self.classes_list[hyper_new]
                    t_new.params_mid_out[edge_list[j]] = [type_new, self.classes_dict[type_new][0],
                                                          np.random.rand(self.classes_dict[type_new][0])]
                else:
                    flat_list = [item for sublist in t_new.params_mid_inp for item in sublist]
                    node = flat_list[edge_list[j] - t_new.n_mid_nodes][0]
                    ind_rand = np.random.randint(len(t_new.params_mid_inp[node]))
                    type_func = t_new.params_mid_inp[node][ind_rand][2]
                    for num, item in enumerate(self.classes_list):
                        if type_func == item:
                            hyper_old = num
                    hyper_new = (hyper_old + np.random.randint(1, 4)) % 4
                    type_new = self.classes_list[hyper_new]
                    t_new.params_mid_inp[node][ind_rand] = [t_new.params_mid_inp[node][ind_rand][0],
                                                            t_new.params_mid_inp[node][ind_rand][1], type_new,
                                                            self.classes_dict[type_new][0],
                                                            np.random.rand(self.classes_dict[type_new][0])]
        return t_new

    def first_order(self, x_input):
        mid_nodes = []  # List of 1-d arrays of shape input.shape[0]
        for mid_inp_edges in self.params_mid_inp:
            sum_mid_inp_edges = 0
            for edge in mid_inp_edges:
                if edge[3] == 2:
                    val = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                              (x, x_input[edge[1]])]).evalf()
                elif edge[3] == 3:
                    val = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                              (a2, edge[4][2]), (x, x_input[edge[1]])]).evalf()
                elif edge[3] == 4:
                    val = self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]), (a2, edge[4][2]),
                                                              (a3, edge[4][3]), (x, x_input[edge[1]])]).evalf()
                sum_mid_inp_edges += val
            mid_nodes.append(sum_mid_inp_edges)
        grads = []
        for i in range(self.dim):
            grad = 0
            for mid_n in range(self.n_mid_nodes):
                if self.mid_inp_edges[mid_n][1][i] == 1:
                    edge = self.params_mid_out[mid_n]
                    if edge[1] == 2:
                        diff_out = self.classes_dict[edge[0]][2].subs([(a0, edge[2][0]), (a1, edge[2][1]),
                                                                       (x, mid_nodes[mid_n])]).evalf()
                    elif edge[1] == 3:
                        diff_out = self.classes_dict[edge[0]][2].subs([(a0, edge[2][0]), (a1, edge[2][1]),
                                                                       (a2, edge[2][2]), (x, mid_nodes[mid_n])]).evalf()
                    elif edge[1] == 4:
                        diff_out = self.classes_dict[edge[0]][2].subs([(a0, edge[2][0]), (a1, edge[2][1]),
                                                                       (a2, edge[2][2]), (a3, edge[2][3]),
                                                                       (x, mid_nodes[mid_n])]).evalf()
                    for item in self.params_mid_inp:
                        for edge in item:
                            if edge[0] == mid_n and edge[1] == i:
                                if edge[3] == 2:
                                    grad_in = self.classes_dict[edge[2]][2].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                                  (x, x_input[i])]).evalf()
                                elif edge[3] == 3:
                                    grad_in = self.classes_dict[edge[2]][2].subs([(a0, edge[4][0]), (a1, edge[4][2]),
                                                                                  (a2, edge[4][2]),
                                                                                  (x, x_input[i])]).evalf()
                                elif edge[3] == 4:
                                    grad_in = self.classes_dict[edge[2]][2].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                                  (a2, edge[4][2]), (a3, edge[4][3]),
                                                                                  (x, x_input[i])]).evalf()
                    grad += diff_out * grad_in
            grads.append(np.mean(grad))
        return grads


    def Kolmogorov_expression(self):
        """
        Here there is a single outer function, the expression is produced by substituting symbol with summation of inners
        expression and then using simplify function.
        This uses approximation of inner functions whereas exact_Kolmogorov_expression uses exact expressions.
        """
        symbols_ = 'X0 '
        for m in range(self.dim - 1):
            if m < self.dim - 2:
                symbols_ += 'X' + str(m + 1) + ' '
            else:
                symbols_ += 'X' + str(m + 1)

        dims_ = symbols(symbols_)
        mid_expr = [None] * self.n_mid_nodes
        for item in self.params_mid_inp:
            expr = sympify(0)
            for edge in item:
                if edge[3] == 2:
                    expr += self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                (x, dims_[edge[1]])])
                elif edge[3] == 3:
                    expr += self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                (a2, edge[4][2]),
                                                                (x, dims_[edge[1]])])
                elif edge[3] == 4:
                    expr += self.classes_dict[edge[2]][1].subs([(a0, edge[4][0]), (a1, edge[4][1]),
                                                                (a2, edge[4][2]), (a3, edge[4][3]),
                                                                (x, dims_[edge[1]])])
            mid_expr[item[0][0]] = expr
        out_expr_ = sympify(0)
        for i in range(self.n_mid_nodes):
            edge = self.params_mid_out[i]
            mid_expr_ = self.classes_dict[edge[0]][1].subs(x, mid_expr[i])
            if edge[1] == 2:
                out_expr_ += mid_expr_.subs([(a0, edge[2][0]), (a1, edge[2][1])])
            elif edge[1] == 3:
                out_expr_ += mid_expr_.subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2])])
            elif edge[1] == 4:
                out_expr_ += mid_expr_.subs([(a0, edge[2][0]), (a1, edge[2][1]), (a2, edge[2][2]), (a3, edge[2][3])])

        return out_expr_