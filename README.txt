Code
====

The code is available at this address: https://github.com/Corendos/MDP_GT_CS7641

Policy Iteration and Value Iteration
====================================

There are two executable file 'App.java' and 'Profiling.java'. The first one solve the problem
and produces the policy image. The second one compute the average compute time for the two algorithms.

App.java
--------

To use the first problem, the line 62 must be 'Map map = smallMap;' and to use the second one, the line
must be 'Map map = largeMap'. There are 3 parameters modifiable:
DISCOUNT_FACTOR: The applied discount factor
MAX_DELTA: The condition on the improvement delta for stopping the algorithm
MAX_ITERATIONS: The maximal number of iterations

Profiling.java
--------------

The same parameters are modifiable here

Q-Learning
==========

The Q Learning scripts use python and require numpy, Pillow and pymdptoolbox.

There is only one script 'qlearning.py' and run the Q-Learning algorithm on the two problems.

There are 3 modifiable parameters, which are really similar to the previous ones:
DISCOUNT_FACTOR: The applied discount factor
ITERATIONS: The number of iterations
MAP: The map to use (smallMap/largeMap)

Other
=====

These algorithms produce output images stored in the 'mdp/output/' directory and have by default the name
of the algorithm used to generate it.