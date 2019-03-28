# Particle-Swarm-Optimization

This project provides code for performing optimization using the particle swarm optimization. A example concerning rosencrock function is shown below.
```
from pylab import *
from scipy.optimize import *
from PSO import PSO
seed(0)
ps = PSO(rosen,random([40,8]),iter_max=1000)
ps.set_intertia_weight('fixed')
ps.set_learning_factor('fixed')
```
