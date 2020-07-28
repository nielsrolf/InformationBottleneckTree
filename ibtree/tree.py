import numpy as np

from ibtree.visualize import optimal_split_plot


def optimal_t_from_selection(x, y, ts, J, min_n):
    """
    Time complexity:
    O(T*N)
    Space complexity:
    O(N+T) - to store the input
    """
    N = len(x)
    best_loss = np.inf
    best_t = -np.inf
    idx = None
    for t in ts: # O(T)
        # evaluate loss for splitting [:s], [s:]
        y_l = y[x <= t] # O(N)
        y_r = y[x > t] # O(N)
        s = len(y_l)
        loss = J(y_l)*s/N + J(y_r)*(N-s)/N # O(N)
        if loss < best_loss and s >= min_n and s <= N-min_n:
            best_loss = loss
            best_t = t
            idx = s
    return best_t, best_loss, idx


def optimal_split(x, y, J, min_n):
    N = len(x)
    ts = np.unique(x) # O(log(N))
    best_t, best_loss, idx = optimal_t_from_selection(x, y, ts, J, min_n)
    return best_t, best_loss


class DecisionTree():
    def __init__(self,
                 J,
                 _prefix="Tree",
                 root=None,
                 min_n=1,
                 max_depth=1000,
                 plot_optimal_split=False,
                 random_projections=False # False or int num_projections
        ):
        self.split_dim = 0
        self.J = J
        self.root = root or self
        self.min_n = min_n
        self.max_depth = max_depth
        self.threshhold = -np.inf
        self.loss = np.inf
        self.leaf_loss = np.inf
        self.left = None
        self.right = None
        self.mean = None
        self.x_mean = None
        self.x_min = None
        self.x_max = None
        self._N = None
        self._prefix = _prefix
        self.n_leaves = 1
        self.depth = 1
        self.plot_optimal_split = plot_optimal_split
        self.rand_id = np.random.rand()
        self.random_projections = random_projections
        self.D = None
        
    def apply_random_projections(self, X):
        if self.D is None:
            N, d = X.shape
            D = np.random.normal(0, 1, size=(d, self.random_projections))
            D = D/np.linalg.norm(D, axis=1, keepdims=True)
            D[:,:d] = np.eye(d)
            self.D = D
        else:
            D = self.D
        return X@D
    
    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def check_input(self, X, y):
        assert (len(X) == len(y) and X.ndim == 2 and y.ndim ==
                2), f"Unexpected shapes: \n  X: {X.shape}\n  y: {y.shape}"
        assert y.sum(1).mean() == 1, "Expected one hot encoded labels"

    def fit(self, X, y):
        """
        Time complexity:
        O(max_num_leaves*D*N^1.5)
        
        Space complexity:
        O(max_depth*N)
        """
        self.check_input(X, y)
        if self.random_projections:
            X = self.apply_random_projections(X)
        self.mean = y.mean(axis=0)
        # Track the own domain for later plotting
        self.mean_x = X.mean(0)
        self.min_x = X.min(0)
        self.max_x = X.max(0)
        #
        self._N = len(X)
        self.leaf_loss = self.J(y)
        N, d = X.shape
        if self.mean.max() == 1 or N < self.min_n or self.max_depth < 1:
            self.loss = self.leaf_loss
            return self
        best_t_idx = None
        for i in range(d): # O(D)
            t, j = optimal_split(X[:, i], y, self.J, self.min_n) # O(N^1.5)
            if j < self.loss:
                self.threshhold, self.loss, self.split_dim = t, j, i
        if self.root.plot_optimal_split:
            optimal_split_plot(X, y, self.J)

        if self.loss < self.leaf_loss:
            self.left = DecisionTree(
                self.J,
                _prefix=f"{self._prefix} and \\ \n    x[{self.split_dim}] <= {self.threshhold}",
                root=self.root,
                min_n=self.min_n,
                max_depth=self.max_depth-1
            )
            self.left.fit(
                X[X[:, self.split_dim] <= self.threshhold],
                y[X[:, self.split_dim] <= self.threshhold]
            )
            self.right = DecisionTree(
                self.J,
                _prefix=f"{self._prefix} and \\ \n    x[{self.split_dim}] > {self.threshhold}",
                root=self.root,
                min_n=self.min_n,
                max_depth=self.max_depth-1
            )
            self.right.fit(
                X[X[:, self.split_dim] > self.threshhold],
                y[X[:, self.split_dim] > self.threshhold]
            )
            self.loss = self.left.loss * (self.left._N / self._N) + \
                self.right.loss * (self.right._N / self._N)
            self.n_leaves = self.left.n_leaves + self.right.n_leaves
            self.depth = max(self.left.depth, self.right.depth) + 1
        else:
            self.loss = self.leaf_loss
        return self

    def get_leaves(self):
        if self.is_leaf:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def predict_single(self, x, max_depth):
        """Works on projected data"""
        if self.is_leaf or max_depth == 0:
            return self.mean
        if x[self.split_dim] <= self.threshhold:
            return self.left.predict_single(x, max_depth=max_depth-1)
        else:
            return self.right.predict_single(x, max_depth=max_depth-1)

    def predict(self, X, max_depth=True):
        if self.random_projections:
            X = self.apply_random_projections(X)
        if max_depth is True:
            max_depth = self.depth
        return np.array([self.predict_single(x, max_depth=max_depth) for x in X])
