from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML, display

from ibtree.loss import entropy, categorical_IB
from ibtree.datasets import grid2d


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


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, x, tree,title):
        self.x = x
        self.generate_data(x, tree)

        # Setup the figure and axes...
        plt.clf()
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_title(title)
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=int(10000/tree.depth),
                                           frames=range(1, tree.depth), init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y = self.x, self.Y[0]
        xmin, xmax = self.x[:,0].min(), self.x[:,0].max()
        ymin, ymax = self.x[:,1].min(), self.x[:,1].max()
        
        self.scat = self.ax.scatter(x[:,0], x[:,1], s=0, animated=True, marker="s", cmap='winter')
        self.ax.axis([xmin, xmax, ymax, ymin]) 
        self.ax.axis('off')

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
    
    def generate_data(self, x, tree):
        # change: method that changes the model or the plane or whatever
        frames_y = []
        for i in range(1, tree.depth):
            y = tree.predict(x, max_depth=i-1)[:,0]
            y[0] = 1-y[0]
            frames_y.append(y)
        self.Y = np.array(frames_y)
            
    def update(self, i):
        # Set x and y data
        self.scat.set_offsets(self.x)
        # Set colors..
        y = self.Y[i-1]
        self.scat.set_array(y)
        self.scat.set_cmap('winter')
        self.scat.set_sizes(0*y+6)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        return display(HTML(self.ani.to_jshtml()))


def animate_fitting(tree, X, y, title, save_as=None):
    tree.fit(X, y)
    X = grid2d(100)
    animation = AnimatedScatter(X, tree, title)
    if save_as:
        animation.ani.save(save_as, writer='imagemagick', fps=1)
    animation.show()
    