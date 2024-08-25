import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from colorama import just_fix_windows_console
def adaptive_sampling(ac,seed3, gp, y_max, bounds, random_state,hyper=[5, 2, 0.78, 0.95,1e-5], dir=None,constraint=None, n_warmup=10000):

    """
    A function to find the minimum of the acquisition function

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param seed3:
        A data used as the initial value for calculating the subproblem.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator.

    :param p_hyper:
        Hyperparameters for the aquisition function.

    :param constraint:
        A ConstraintModel.

    :param n_warmup:
        number of times to randomly sample the acquisition function.

    :param n_iter:
        number of times to run scipy.minimize.

    Returns
    -------
    :return: 

    x_max, The arg max of the acquisition function.

    PFC, The probability that the constraint is satisfied at x_max.

    UCB_obj, The predicted value of the objective function calculated by the UCB function at x_max.

    max_acq, The value of the acquisition function at x_max.
    """
    def acquisition(x, m1=5, m2=2, p1=0.78, p2=0.95,s=1e-5, constraint=constraint):
        x_reshaped = x.reshape(-1, x.shape[-1])
        p_cons = constraint.predict(x_reshaped)
        ac_predict = -ac(x_reshaped, gp=gp, y_max=y_max)
        ac_value = np.zeros((len(x_reshaped)))
        mask1 = p_cons < p1
        mask2 = (p1 <= p_cons) & (p_cons < p2)
        mask3 = p2 <= p_cons
        ac_value[mask1] = ac_predict[mask1] + 1 / (s + p_cons[mask1] ** m1)
        ac_value[mask2] = ac_predict[mask2] + 1 / (p_cons[mask2] ** m2)
        ac_value[mask3] = ac_predict[mask3]
        return ac_value
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    sampler = LatinHypercubeSampler(bounds,random_state)
    sample_LHS = sampler.sample(n_warmup)
    if constraint == None:
        ys = ac(x_tries, gp=gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(10, bounds.shape[0]))
    else:
        ys = acquisition(sample_LHS,m1=hyper[0],m2=hyper[1],p1=hyper[2],p2=hyper[3],s=hyper[4],constraint=constraint)
        x_max = sample_LHS[ys.argmin()]
        max_acq = ys.min()
        argmin_x = (ys).argsort()[:6]
        x_seeds1 = random_state.uniform(bounds[:, 0], bounds[:, 1],size=(6, bounds.shape[0]))
        x_seeds2 = sample_LHS[argmin_x]
        x_seeds3 = seed3
        if x_seeds3.size > 0:
            x_seeds = np.concatenate((x_seeds1,x_seeds2,x_seeds3),axis=0)
        else:
            x_seeds = np.concatenate((x_seeds1,x_seeds2),axis=0)
    if constraint is not None:
        def to_minimize(x):
            ys = acquisition(x,m1=hyper[0],m2=hyper[1],p1=hyper[2],p2=hyper[3],s=hyper[4],constraint=constraint)
            return ys
    else:
        to_minimize = lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max)

    for x_try in x_seeds:
        # Find the minimum the acquisition function
        res = minimize(lambda x: to_minimize(x),
                       x_try,
                       bounds=bounds,
                       method="SLSQP")
        # See if success
        if not res.success:
            continue
        # Store it if better than previous minimum(maximum).
        if constraint is not None:
            if max_acq is None or np.squeeze(res.fun) <= max_acq:
                x_max = res.x
                max_acq = np.squeeze(res.fun)
        else:
            if max_acq is None or -np.squeeze(res.fun) >= max_acq:
                x_max = res.x
                max_acq = -np.squeeze(res.fun)
    print('maxmax',constraint.predict(x_max.reshape(1, -1)),-ac(x_max.reshape(1, -1), gp=gp, y_max=y_max),max_acq,x_max)
    return np.clip(x_max, bounds[:, 0], bounds[:, 1]),constraint.predict(x_max.reshape(1, -1)),-ac(x_max.reshape(1, -1), gp=gp, y_max=y_max),max_acq

class LatinHypercubeSampler:
    def __init__(self, bounds, random_state=None):
        self._bounds = bounds
        self.random_state = random_state
        
    def lhs_sample(self, n_samples=1):
        """
        Generate samples using Latin Hypercube Sampling (LHS).

        Parameters:
        n_samples (int): Number of samples to generate.

        Returns:
        numpy.ndarray: LHS samples of shape (n_samples, self.dim).
        """
        
        if isinstance(self._bounds, dict):
            bounds_values = list(self._bounds.values())
        elif isinstance(self._bounds, np.ndarray):
            bounds_values = self._bounds
        n_dimensions = len(bounds_values)
        # Initialize an empty array for LHS
        result = np.empty((n_samples, n_dimensions))
        
        for i in range(n_dimensions):
            lower, upper = bounds_values[i]
            # Generate LHS intervals
            interval = np.linspace(lower, upper, n_samples + 1)
            points = (interval[:-1] + interval[1:]) / 2
            self.random_state.shuffle(points)
            result[:, i] = points
        return result

    def sample(self,n_samples):
        # Assuming n_samples=1 for single sample generation
        lhs_data = self.lhs_sample(n_samples=n_samples)
        return lhs_data

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.

    kind: {'ucb', 'ei', 'poi'}
        * 'ucb' stands for the Upper Confidence Bounds method
        * 'ei' is the Expected Improvement method
        * 'poi' is the Probability Of Improvement criterion.

    kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is
            the highest.

    kappa_decay: float, optional(default=1)
        `kappa` is multiplied by this factor every iteration.

    kappa_decay_delay: int, optional(default=0)
        Number of iterations that must have passed before applying the
        decay to `kappa`.

    xi: float, optional(default=0.0)
    """

    def __init__(self, kind='ucb', kappa=2.5, xi=0, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


class NotUniqueError(Exception):
    """A point is non-unique."""
    pass


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                        constraint_value=iteration["constraint"] if optimizer.is_constrained else None
                    )
                except NotUniqueError:
                    continue

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)


just_fix_windows_console()
