from matplotlib import pyplot as plt
import numpy as np

from ibtree.loss import entropy, categorical_IB


def optimal_split_plot(X, y, J):
    """Plots the loss as function of the threshhold for each dimension
    of X

    Args:
        X (np.array(shape=[N, 2])): Input data for the tree
        y (np.array(shape=[N, C])): One-hot encoded target for X
        J (impurity_measure: Y -> float): Loss function, e.g. entropy or categorical_IB
    """
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    N, d = X.shape
    Y = y
    for i in range(d):
        x = X[:, i]
        sort = np.argsort(x)
        x, y_i = x[sort], y[sort]
        best_loss = np.inf
        best_split_idx = 0
        h = []
        eval_treshhold = np.unique(x)
        for t in eval_treshhold:
            # evaluate loss for splitting [:s], [s:]
            y_l = y_i[x <= t]
            y_r = y_i[x > t]
            s = len(y_l)
            loss = J(y_l)*s/N + J(y_r)*(N-s)/N
            h.append(loss)

        plt.plot(eval_treshhold, h, label=f"x{i}", color=c[i])

    plt.xlabel("Threshhold")
    plt.ylabel("Loss")
    plt.legend()


def plot_on_grid(y, N):
    """Visualize a toy dataset, where X is a (N, N)-grid and y = f(X)
    """
    plt.imshow(y.reshape((N, N)), cmap='winter', interpolation='nearest')
    plt.xlabel("x1")
    plt.ylabel("x0")
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)


def plot_entropy_of_bernoulli():
    """Plot the entropy of a Bernoullie distribution
    """
    p = []
    h = []
    for t in range(1000):
        p += [t/1000]
        y = np.zeros((1000, 2))
        y[:t, 0] = 1
        y[t:, 1] = 1
        h += [entropy(y)]

    plt.plot(p, h)
    plt.xlabel("t")
    plt.ylabel("H")
    plt.title("Entropy of Bernoulli_t")
    plt.show()


def print_node(self, prefix=""):
    """Helper for the function `print_tree` below

    Args:
        self: DecisionTree
        prefix (str, optional): Indentation for the node. Defaults to "".
    """ 
    print(f"{prefix} >> N: {self.mean}")
    print(f"{prefix} >> N: {self._N}")
    print(f"{prefix} >> J: {self.leaf_loss:.2f}")
    print(f"{prefix} >> Acc: {self.mean.max():.2f}")


def print_tree(self, prefix=""):
    """Recursively prints the decision nodes of a tree.

    Args:
        self: DecisionTree
        prefix (str, optional): Indentation for the node. Defaults to "".
    """ 
    if self.is_leaf:
        self.print_node(prefix)
    else:
        print(
            f"{prefix} X_{self.split_dim} <= {self.threshhold} : {self.left._N} vs {self._N-self.left._N}")
        self.left.print_tree(prefix+"  |")
        if self.right is not None:
            print(f"{prefix} X_{self.split_dim} > {self.threshhold}")
            self.right.print_tree(prefix+"  |")