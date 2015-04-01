#Swarm Optimization

##Motivation
So far we have been quite hands-on with reading data in with pandas.

This morning we will step away from using data, instead we will be
testing the performance of Swarm Optimization using a standard
function. This is an _**important**_ skill to have since you can quickly
gauage performance of an optimization algorithm before applying your it
on any data at scale. Furthermore the function we picked is hard
to optimize for, so if the algorithm does well on that, you can be confident
it will do well across the board.

##Griewank Function

[Griewank](http://mathworld.wolfram.com/GriewankFunction.html) is a
standard function to gauge performance of optimization algorithms
because it has a lot of noise and therefore local minima (_see Figure 1_).
The global optimal is at 0 in a 1-dimensional space. There is a local
minimum every 6.28005 interval (indicated by the red dots in _Figure 1_).

**Today we will try to implement Swarm Optimization to find
  that global minimum at 0.**

<div align="center"><img src="/imgs/griewank.png"></div>
<p align="center"><b><i>Figure 1. Image of the Griewank Function</i></b></p>

There are more standard functions you can test your algorithm with as
described in the paper [_**Dynamic Sociometry in Particle Swarm Optimization**_](/readings/)
in the _**readings**_ folder.


##Step 1: Build Agent Class (agent.py)

**Agents are the base units of a swarm. They carry parameters that can be
updated.**

<br>

&nbsp;&nbsp;1\.&nbsp;&nbsp;I defined the attributes in the ```__init__()``` for you.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Read the following and get an idea what
 the attributes are.

```
params (Numpy Array)
    - Parameters that each agent carries. Can be more than 1 dimension
    - Updated at each iteration.


swarm_class (Swarm Class Object)
    - The Swarm Class we are going to instantiate later
    - Yes, a Class can be passed in as an argument into another class
```

<br>
&nbsp;&nbsp;2\.&nbsp;&nbsp;Implement ```update_params``` by filling in the
code stub in ```agent.py```.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It should have the following functionalities.

```
- Calculate the difference between the lowest cost paramters
  (self.swarm_class.best_params)
  and the parameters of the current agent (self.params)

- Factor the difference by a learning factor
  (self.swarm_class.learning_factor)

- Update self.params by adding to the factored difference
```

##Step 2: Build Swarm Class (swarm_basic.py)

**A Swarm contains Agents and is able to dictate the position / paramter
of each Agent**

<br>

1. Here we are going to build a version of Swarm Optimization where at each iteration we determine the agent with
   the lowest cost, and subsequently
   all the other agents will move the direction of the agent with the lowest
   cost. See the following flowchart to understand the step-by-step flow of
   the algorithm.

<div align="center"><img src="/imgs/swarm_flow.png"></div>


&nbsp;&nbsp;2\.&nbsp;&nbsp;&nbsp;Again I have given you the attributes of the class.
It is a long list, so I have put the descriptions in 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```swarm_basic.py```.
Read them and understand what knobs can be tweaked in the algorithm.
We have covered  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;them more or less in the lecture. If there is anything
unclear, give a shout out. I will be happy to explain.

&nbsp;&nbsp;3\.&nbsp; Fill the function ```initialize_swarm```. You would need to
initialize parameters randomly within the specified &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bounds ```self.bounds```.
Even though we are dealing with a 1-dimensional parameter, we have to use
an array. In &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;most 
future usage, you will to deal with multi-dimensional
data. I have given you ```draw_random_params``` as a &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;helper
function if you want to break things up.

&nbsp;&nbsp;4\.&nbsp; The next step is to define the cost function. I have implemented the
```griewank``` function for you in 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```cost_functions.py```.
Fill in ```get_cost_function``` in ```swarm_basic.py```
to retrieve the cost function from the 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```CostFunctions Class```.
Assign the function to the ```self.cost_func``` variable in Swarm.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Hint: Use** ```CostFunctions().griewank```

&nbsp;&nbsp;5\.&nbsp; Now we need a function that takes care of iteratively reassigning
agent parameters, i.e. ```step 2``` to ```step 5``` of the 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;flow chart. Implement
one cycle by filling in ```one_iteration```. See below for hints.

```
- Loop through all the agents and get list of costs using self.cost_func

- Get the minimum cost and the paramaters associated
  (Assign to self.best_params)
  I have provided with find_lowest_cost_index to help

- Loop through the agents again and call agent.update_params().
  You have already implemented update_params in agent.py in Step 1.
  Go back to update_params if you are unsure how the flow works.
```

&nbsp;&nbsp;6\.&nbsp; You might also want to keep track of the mean cost at each iteration
    Fill in ```find_mean_cost``` for that and 
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;include it in ```one iteration```

&nbsp;&nbsp;7\.&nbsp; Stitch everything up in ```fit```. It should include everything
    in the flowchart, and iterate for ```self.iter_stop``` times.

&nbsp;&nbsp;8\.&nbsp; When you are done, run ```python swarm_basic.py```. If your best parameter
is a multiple of 6, then you are stuck in a 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;local minimum. There is just one
little thing in ```Step 9``` that you need to do.

&nbsp;&nbsp;9\.&nbsp; Go back to ```agent.py```, tweak ```update_params``` to allow straying of agent
in opposite direction at probability lower &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or equal to stray rate

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Hint: Use** ```self.swarm_class.stray_rate```

&nbsp;&nbsp;10\.&nbsp; Run ```python swarm_basic.py``` again, you should get a best parameter of
0
