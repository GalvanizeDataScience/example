import numpy as np
import numpy.random as npr


class Agent(object):
    """Agents that are to be contained in a list(swarm) in swarm class"""
    def __init__(self, params, swarm_class):
        self.params = params
        self.cost = 1e50
        self.swarm_class = swarm_class

    def __str__(self):
        """Print the current position"""
        return 'Current position: ', self.params

    def update_params(self):
        """Based on best params determined in swarm class,
        update params of the agent"""
        # Calculate step size and if stray then go in the opposite direction
        params_delta = self.swarm_class.best_params - self.params
        step_size = params_delta * self.swarm_class.learning_rate
        if npr.uniform() <= self.swarm_class.stray_rate:
            self.params -= step_size
        else:
            self.params += step_size
