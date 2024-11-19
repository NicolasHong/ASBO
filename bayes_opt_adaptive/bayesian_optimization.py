import warnings

from .constraint import ConstraintModel

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, ensure_rng,adaptive_sampling,LatinHypercubeSampler
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np 

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """

    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)

class BayesianOptimization(Observable):
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: A ConstraintModel. Note that the names of arguments of the
        constraint function and of f need to be the same.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    relax_margin: int or float, optional(default=0)
        If the black box type is process simulation software, in order to
        avoid problems caused by numerical accuracy, the constraint 
        variables can be relaxed..

    LHS:  bool, optional (default=True)
        If True, The Latin hypersquare sampling function will be used 
        when selecting the initial point.
        
    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    allow_duplicate_points: bool, optional (default=False)
        If True, the optimizer will allow duplicate points to be registered.
        This behavior may be desired in high noise situations where repeatedly probing
        the same point will give different answers. In other situations, the acquisition
        may occasionaly generate a duplicate point.

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """

    def __init__(self,
                 f,
                 pbounds,
                 constraint=None,
                 random_state=None,
                 relax_margin=0,
                 LHS = True,
                 verbose=2,
                 bounds_transformer=None,
                 allow_duplicate_points=False):
        self._random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self._queue = Queue()
        self._pbounds = pbounds
        self._LHS = LHS
        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )
        self.columns = {}
        self.list_tp, self.list_ucb, self.list_ac = [], [], []

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(f, pbounds, random_state=random_state,
                                      allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun,
                constraint.lb,
                constraint.ub,
                random_state=random_state,
                relax_margin=relax_margin
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state
            )
            self.is_constrained = True

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)
    @property
    def space(self):
        return self._space

    @property
    def constraint(self):
        if self.is_constrained:
            return self._space.constraint
        return None

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target, constraint_value=None):
        """Expect observation with known target"""
        self._space.register(params, target, constraint_value)
        
        self.dispatch(Events.OPTIMIZATION_STEP)


    def probe(self, params, lazy=True):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """

        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)
    @staticmethod
    def adaptive_sampling_parallel(utility_function, seed3, gp, constraint, space, random_state, p_hyper):
        suggestion, sub_cons, sub_acq, sub_obj = adaptive_sampling(
            ac=utility_function.utility,
            seed3=seed3,
            gp=gp,
            constraint=constraint,
            y_max=space.target.max(),
            bounds=space.bounds,
            random_state=random_state,
            hyper=p_hyper,dir=dir
        )
        return suggestion, sub_cons, sub_acq, sub_obj

    def suggest(self, utility_function,p_hyper,dir,strategy):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            if self.is_constrained:
                self.constraint.fit(self._space.params,
                                    self._space._constraint_values)
        seed3 = self._space.seeds3

        if len(p_hyper)==1:
            # Finding argmin of the acquisition function.
            suggestion,sub_cons,sub_ucb,sub_maxac = adaptive_sampling(ac=utility_function.utility,
                                seed3=seed3,
                                gp=self._gp,
                                constraint=self.constraint,
                                y_max=self._space.target.max(),
                                bounds=self._space.bounds,
                                random_state=self._random_state,
                                hyper=p_hyper[0],dir=dir,strategy=strategy)
            if dir !=None:
                suggestion_length = len(suggestion)
                for i in range(suggestion_length):
                    column_name = f'x{i+1}'
                    if column_name not in self.columns:
                        self.columns[column_name] = []
                    self.columns[column_name].append(suggestion[i])
                
                # Append other values that are not dynamically generated
                self.list_tp.append(sub_cons[0])
                self.list_ucb.append(sub_ucb[0])
                self.list_ac.append(sub_maxac)
                
                # Add fixed columns to the DataFrame dictionary
                self.columns['tp'] = self.list_tp
                self.columns['ucb'] = self.list_ucb
                self.columns['ac'] = self.list_ac

                # Create the DataFrame with dynamically generated columns
                df = pd.DataFrame(self.columns)
                df.to_csv(dir, index=False)    
            return self._space.array_to_params(suggestion)
        else:
            process_num = len(p_hyper)
            # print('Compute the objective function in parallel, and the number of parallel runs is:',process_num)
            from multiprocessing import Pool
            with Pool(process_num) as p:
                results = p.starmap(BayesianOptimization.adaptive_sampling_parallel, [
                    (utility_function, seed3, self._gp, self.constraint, self._space, self._random_state, ph) 
                    for ph in p_hyper
                ])
            suggestions = [result[0] for result in results]
            return [self.space.array_to_params(suggestions[i]) for i in range(len(suggestions))]

            # return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points, init_points_array):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        if self._LHS ==True and type(init_points_array) is not np.ndarray:
            sampler = LatinHypercubeSampler(self._pbounds, self._random_state)
            sample = sampler.sample(init_points)
            for _s in sample:
                self._queue.add(_s)
        elif type(init_points_array) is np.ndarray:
            for _s in init_points_array:
                self._queue.add(_s)
            if init_points_array.shape[0] < init_points:
                sampler = LatinHypercubeSampler(self._pbounds, self._random_state)
                sample = sampler.sample(init_points-init_points_array.shape[0])
                for _s in sample:
                    self._queue.add(_s)                
        else:
            for _ in range(init_points):
                self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose, self.is_constrained)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 init_points_array=None,
                 n_iter=25,
                 p_hyper=[4, 2, 0.78, 0.95,1e-3],
                 acquisition_function=None,
                 dir = None,
                 strategy = 123,
                 acq=None,
                 kappa=None,
                 kappa_decay=None,
                 kappa_decay_delay=None,
                 xi=None,
                 **gp_params):

        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acquisition_function: object, optional
            An instance of bayes_opt.util.UtilityFunction.
            If nothing is passed, a default using ucb is used

        All other parameters are unused, and are only available to ensure backwards compatability - these
        will be removed in a future release
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points,init_points_array)

        if acquisition_function is None:
            util = UtilityFunction(kind='ucb',
                                   kappa=2.5,
                                   xi=0.0,
                                   kappa_decay=1,
                                   kappa_decay_delay=0)
        else:
            util = acquisition_function

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util,p_hyper,dir,strategy)
                iteration += 1
            self.probe(x_probe, lazy=False)
            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters to the internal Gaussian Process Regressor"""
        self._gp.set_params(**params)
