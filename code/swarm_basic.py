from agent import Agent
from cost_functions import CostFunctions
import numpy.random as npr
import numpy as np


class Swarm(object):
    """
    Attribute Descriptions:

    cost_func_name (String)
        - Name of the cost function. We will do 'griewank' to start with
        - Used to retrieve cost function callable in the CostFunctions Class

    bounds (List of List)
        - Each sublist has upper and lower bound of the feature
        - I will give you bounds later, don't worry
        - Now just build the class as if you have all the attributes

    swarm_size(Integer: Optional)
        - Default to 200
        - Empirically shown to be good for most problems

    iter_stop(Integer: Optional)
        - Default to 100 iterations before terminating the algorithm
        - Will be sufficient if you implemented it correctly

    stray_rate(Float: Optional)
        - Default to .05
        - 5 percent of the time, it takes random step instead of going to the
        lowest cost agent
    """

    def __init__(self, cost_func_name, bounds,
                 swarm_size=200, iter_stop=2000,
                 stray_rate=.05, learning_rate=.1,
                 verbose=2):
        self.cost_func_name = cost_func_name
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iter_stop = iter_stop
        self.stray_rate = stray_rate
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Contains all the agents initialized
        self.swarm = []

        # Callable cost function. INPUT: NP_ARRAY(params), OUTPUT: FLOAT(cost)
        self.cost_func = None

        # Keep track of all the currently caluclated costs
        # Emptied on starting every round
        self.cost_lst = []

        # The params that give the lowest cost in each round
        self.best_params = np.array([])

        # Keep track of the mean cost and lowest cost each iteration
        self.mean_cost = 1e50
        self.lowest_cost = 1e50

    def draw_random_params(self):
        """INPUT:
        - self.bounds (List: Instance variable) [parameter bounds]

        OUTPUT:
        - (numpy array) [random paramter array within bounds]

        Given the bounds of each feature, generate a random param"""
        pass

    def initialize_swarm(self):
        """INPUT:
        - self.self.swarm_size(Integer: Instance Variable)

        OUTPUT:
        - None

        Make the swarm of agents according to self.swarm_size
        Assign to self.swarm
        """
        pass

    def get_cost_function(self):
        """INPUT:
        - self.cost_func_name(String: Instance Variable) [e.g. 'griewank']

        OUTPUT:
        - None

        Use self.cost_func_name to get cost_func from CostFunctions
        (cost_functions.py). Assign callable to self.cost_func
        """
        pass

    def print_output(self, i):
        """Function to report output. Use this if you want to"""
        print 'Iteration: ', i
        print '  Best Params: ', self.best_params
        print '  Mean Cost: ', self.mean_cost
        print '  Lowest Cost: ', self.lowest_cost

    def one_iteration(self):
        """INPUT
        - self.swarm(List: Instance variable)
        - self.cost_lst(List: Instance variable)
        - self.cost_func(Callable: Instance variable)

        OUTPUT:
        - self.best_param(Numpy Array: Instance variable)
        - self.lowest_cost(Float: Instance variable)

        Running one iteration of swarm. Calculate cost and
        updating positions of Agents
        """
        pass

    def find_lowest_cost_index(self):
        """INPUT:
        - self.cost_lst(List: Instance variable)

        OUTPUT:
        - lowest_ind(Integer) [Index of minimum cost agent]

        Get the minimum cost index in self.cost_lst at this iteration
        """
        pass

    def find_mean_cost(self):
        """INPUT:
        - self.cost_lst(List: Instance variable)

        OUTPUT:
        - mean(Float) [Mean cost]

        Get the mean cost in self.cost_lst at this iteration
        """
        pass

    def fit(self):
        """Run the algorithm and stop when the iterations are up"""
        pass

if __name__ == '__main__':
    griewank_swarm = Swarm('griewank', [[-600, 600]],
                           iter_stop=100, verbose=1)
    griewank_swarm.fit()
