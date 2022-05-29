import numpy as np
import symbolic_metamodeling


def model_func(x):
    return np.exp(3 * x[:, 0] + x[:, 1])


n = 100
X_train = np.random.rand(n, 2)

metamodel = symbolic_metamodeling.symbolic_metamodel(model_func, X_train, "function")

metamodel.fit(num_iter=10, batch_size=X_train.shape[0], learning_rate=.01)
