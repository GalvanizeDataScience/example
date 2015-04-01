from agent import Agent
from cost_functions import CostFunctions
import numpy.random as npr
import numpy as np


class Swarm(object):
    """INPUT:
    - cost_func_name(STR)
        Name of the cost function. Now only 'griewank' implemented.

    - bounds(DICT)
        Each sublist has upper and lower bound of the feature.

    - swarm_size(INT: OPT)
        Default to 200. Empirically shown to be good for most problems

    - iter_stop(INT: OPT)
        Default to 2000. Stop regardless of other condition when the number
        of iterations are reached

    - stray_rate(FLOAT: OPT)
        Default to .05. 5 percent of the time, it takes random step

    - verbose(INT:OPT)
        Default to 1. How frequent the output is printed. 1/2/3"""

    def __init__(self, cost_func_name, bounds,
                 swarm_size=200, iter_stop=2000,
                 stray_rate=.05, learning_rate=.1,
                 lowest_spawn=10, lowest_spawn_search=.05,
                 verbose=1):
        self.cost_func_name = cost_func_name
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iter_stop = iter_stop
        self.stray_rate = stray_rate
        self.learning_rate = learning_rate
        self.lowest_spawn = lowest_spawn
        self.lowest_spawn_search = lowest_spawn_search
        self.verbose = verbose

        # Contains all the agents initialized
        self.swarm = []
        # Callable cost function.
        # INPUT: NP_ARRAY(params), OUTPUT: FLOAT(cost)
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
        """Given the bounds of each feature, generate a random position"""
        params_lst = []
        for bound_lst in self.bounds:
            lower = bound_lst[0]
            upper = bound_lst[1]
            draw = npr.uniform(lower, upper)
            params_lst.append(draw)
        return np.array(params_lst)

    def spawn(self, num_agents):
        """INPUT:
        - n(INT) [The number of agents to be spawned]
        Spawn agents and pushing into list (inst var swarm_lst)
        """
        for _ in range(num_agents):
            init_pos = self.draw_random_params()
            self.swarm.append(Agent(init_pos, self))

    def initialize_swarm(self):
        """Make the swarm of agents according to a predefined number"""
        print 'Initializing Swarm...'
        self.spawn(self.swarm_size)

    def get_cost_function(self):
        """Get callable function from a string name used to instantiate
        the Swarm class"""
        print 'Getting Cost Function'
        cost_func_inst = CostFunctions()
        if self.cost_func_name == 'griewank':
            self.cost_func = cost_func_inst.griewank
        else:
            raise Exception('Cost function not implemented. Use "griewank"')

    def print_output(self, i):
        """Function to report output"""
        print 'Iteration: ', i
        print '  Best Params: ', self.best_params
        print '  Mean Cost: ', self.mean_cost
        print '  Lowest Cost: ', self.lowest_cost

    def fit(self):
        """Run the algorithm and stop when the iterations are up"""
        # Retrieve cost function from Cost class (self.cost_func)
        self.get_cost_function()
        # Define the swarm of agents
        self.initialize_swarm()

        print 'Starting to run Optimization...'
        # While stop condition not met, iterate
        for i in range(self.iter_stop):
            self.one_iteration()
            if self.verbose:
                if self.verbose == 1:
                    if i % 200 == 0:
                        self.print_output(i)
                elif self.verbose == 2:
                    if i % 10 == 0:
                        self.print_output(i)
                elif self.verbose == 3:
                    self.print_output(i)
        print 'Done! You best paramter is:', self.best_params
        print 'Estimated Lowest cost: ', self.lowest_cost

    def one_iteration(self):
        """Running one iteration of swarm.
        Calculate cost and updating positions of Agents"""
        # Empty cost list
        if self.cost_lst:
            del self.cost_lst[:]

        # Calculate cost for each agent
        for agent in self.swarm:
            cost = self.cost_func(agent.params)
            self.cost_lst.append(cost)

        # Get the lowest cost in the current iteration
        self.mean_cost = self.find_mean_cost()
        lowest_cost_ind = self.find_lowest_cost_index()
        self.best_params = self.swarm[lowest_cost_ind].params
        self.lowest_cost = self.cost_lst[lowest_cost_ind]

        # Update position of each agent based on best params and lowest_cost
        for agent in self.swarm:
            agent.update_params()
            # The lowest agent search for lower cost nearby
            self.lowest_agent_explore(lowest_cost_ind)

    def bound_ranges(self):
        """Ranges of each feature"""
        return np.array([upper - lower for upper, lower in self.bounds])

    def draw_epsilon(self, params_epsilon):
        """Randomly decide how much the lowest agent explores within
        predefined bounds"""
        rand_eps_lst = []
        for epsilon in params_epsilon:
            if epsilon > 0:
                rand_epsilon = npr.uniform(0, epsilon)
            else:
                rand_epsilon = npr.uniform(epsilon, 0)
            rand_eps_lst.append(rand_epsilon)
        return np.array(rand_eps_lst)

    def check_within_bounds(self, params):
        """Check if explored values are within param bounds"""
        for i in range(len(params)):
            lower = self.bounds[i][0]
            upper = self.bounds[i][1]
            return lower <= params[i] and upper >= params[i]

    def lowest_agent_explore(self, lowest_cost_ind):
        """Spawn random agents around the lowest agent.
        If any of those have lower cost, return the more desirable params"""
        bound_range_arr = self.bound_ranges()
        spawned = 0
        near_best_costs = []
        while spawned < self.lowest_spawn:
            params_epsilon = bound_range_arr * \
                             self.lowest_spawn_search
            rand_epsilon = self.draw_epsilon(params_epsilon)
            near_best_params = self.best_params + rand_epsilon
            within_bounds = self.check_within_bounds(near_best_params)
            if within_bounds:
                spawned += 1
                near_best_costs.append(self.cost_func(near_best_params))
        spawned_min_cost = min(near_best_costs)

        # Reassign the lowest agent params and update lowest cost
        if spawned_min_cost < self.lowest_cost:
            self.swarm[lowest_cost_ind].params = near_best_params
            self.lowest_cost = spawned_min_cost

    def find_lowest_cost_index(self):
        """Get the minimum cost in the current iteration"""
        cost_array = np.array(self.cost_lst)
        lowest_ind = cost_array.argmin()
        return lowest_ind

    def find_mean_cost(self):
        """Get the mean cost in the current iteration"""
        return np.mean(self.cost_lst)

if __name__ == '__main__':
    griewank_swarm = Swarm('griewank', [[-600, 600]],
                           iter_stop=100, verbose=1,
                           lowest_spawn_search=.1, lowest_spawn=20)
    griewank_swarm.fit()
