#!/usr/bin/env python3

"""粒子群算法"""

__version__ = '1.1.1'

import numpy as np
from numpy import iterable
import numpy.random
try:
    from scipy.optimize import OptimizeResult
except ImportError:                       # Earlier version of scipy
    from scipy.optimize import Result
    OptimizeResult = Result

    
class particle:
    "A bird."
    def __init__(self,fun:'callable, objective function',
                 x0:'initial position',
                 v0:'initial speed',
                 w:'inertia weight'=0.5,
                 c1:'self cognition'=2,
                 c2:'social experience'=2,
                 vmax:'max speed'=None):
        """if vmax is not none, then the the speed will be normalized if the norm of speed is larger than vmax(evaluate by norm_inf)"""
        self.f = fun            # objective function

        self.x = x0             # position, const void *
        self.v = v0             # speed
        self.x_opt = x0 # The best position in history 本粒子的历史最优点
        self.y_opt = fun(self.x_opt) # the value of point x_opt
        
        self.w = w              # inertia weight
        self.c1 = c1            # self cognition
        self.c2 = c2            # social experience

        self.vmax = vmax        # maximal speed
        
    def set_coeff(self,w=None,c1=None,c2=None):
        """set new values for the coefficients"""
        if w != None:
            self.w = w
        if c1 != None:
            self.c1 = c1
        if c2 != None:
            self.c2 = c2
    
    def set_neighbour(self,nbs:'list of neighbours'):
        """设置该粒子的邻域。
        若此粒子也在自己的邻域中，则应将其加入到nbs中。"""
        if len(nbs) < 1:
            raise ValueError("number of neighbours should not be less than 1")
        self.nbs = nbs
    
    def get_opt_from_neighbours(self):
        """寻找邻域中的最优粒子"""
        p = min(self.nbs,key=lambda p:y_opt)
        return p

    def iter(self,p_opt:'The best particle of all particles'=None):
        """Do the itertion once.
        If pg is not given, it will be searched from the neighbours."""
        if p_opt == None:
            p_opt = self.get_opt_from_neighbours()
        self.v *= self.w                         # 惯性部分
        self.v += self.c1*np.random.random()*(self.x_opt-self.x) # congnition自我认知部分
        self.v += self.c2*np.random.random()*(p_opt.x-self.x) # social 社会经验部分
        if self.vmax:
            vmax = max(abs(self.v))
            if self.vmax < vmax:
                self.v *= self.vmax / vmax
        self.x = self.x + self.v # content in self.x MUST be READ_ONLY

        y = self.f(self.x)
        if y < self.y_opt:
            self.x_opt = self.x
            self.y_opt = y

            
class PSO:
    """find the minimal value of a function using the Particle Swarm Optimization"""
    def __init__(self,fun:'callable, objective function, or a list containing functions',
                 x:'initial value, lists like [x0,x1,...]',
                 v:'initial speed of every particle, will be auto setted if None is given' = None,
                 iter_max:'maximal number of iterations' = 100,
                 fun2:'will be called after each iteration with argment:best_x,fun(best_x),niter'=None,
                 tol:'tolerance' = 0):
        """if any of fun,x or v0 is a list, their size should be the same. And n should be their size if given."""
        self._fun = fun
        self._x = np.array(x,dtype=float)
        self._n = len(x)      # 粒子数
        self._v = v
        self._iter_max = iter_max
        self._fun2 = fun2
        self._tol = tol
        self._niter = 0      # 迭代次数
        self._swarm = []      # 记录各个粒子
        self._p_opt = None    # 取得最优值的粒子
        self._intertiaWeight = {}
        self.set_intertia_weight('fixed')
        self._learningFactor = {}
        self.set_learning_factor('fixed')
        self._msg = ''

    @property
    def fun(self):
        return self._fun
    @fun.setter
    def fun(self,fun,noerror=False):
        """Set the objective function.
        fun can be a function or a list of functions. If fun is a list, it should have the same size of the number of particles"""
        if self.niter and not noerror:
            import sys
            sys.stderr.write("fun should not be changed after iteration started.\n"
            "However, you can use noerror=True to do change the functions.")
            raise AttributeError("fun can not be setted after iteration started")
        self._fun = fun
        # 若已经开始迭代，则需要重新设置各个粒子的目标函数
        if niter:
            if iterable(self.fun):
                assert len(self.fun==self.n)
            else:
                self._fun = [self.fun] * self.n
            for i in range(len(self._swarm)):
                self._swarm[i].f = self._fun[i]
        

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self,x):
        """Set the initial value.
        x is the initial value (position of particles), shoule be range like [[x0_lower,x0_upper],[x1_lower,x1_upper],...] or lists like [x0s,x1s,...]"""
        if self.niter:
            raise AttributeError("x can not be setted after iteration started")
        self._x = np.array(x,dtype=float)

    @property
    def n(self):
        """The number of particles"""
        return self._n

    @property
    def v(self):
        return self._v
    @v.setter
    def v(self,v):
        """Set the initial speed.
        v is the initial speed of particles, shoule be a list of speeds or None"""
        if self.niter:
            raise AttributeError("v can not be setted after iteration started")
        if v != None:
            self._v = np.array(v,dtype=float)
        else:
            self._v = None

    @property
    def iter_max(self):
        return self._iter_max
    @iter_max.setter
    def iter_max(self,n,noerror=False):
        """set the maximal number of iterationgs"""
        if self.niter and not noerror:
            import sys
            sys.stderr.write("Iter_max should not be changed after iteration started.\n"
            "However, you can use noerror=True to do change the value.")
            raise AttributeError("iter_max can not be setted after iteration started")
        self._iter_max = int(n)

    @property
    def tol(self):
        return self._tol
    @tol.setter
    def tol(self,tol):
        if self.niter:
            raise AttributeError("tolerance can not be setted after iteration started")
        self._tol = float(tol)
    
    @property
    def niter(self):
        return self._niter

    def get_ready(self):
        """Getting ready for the iteration"""
        if self.niter:
            raise AttributeError("Calling get_ready after iter started!\n")
        # 目标函数
        if iterable(self.fun):
            assert len(self.fun)==self.n
        else:
            self._fun = [self.fun] * self.n
        # 初速度
        if self.v != None:
            assert len(self.v) == self.n
        else:
            v_max = self.x.max(0) - self.x.min(0)
            self._v = np.array([v_max]*self.n,dtype=float)
            self._v *= np.random.random(self.x.shape) - 0.5
        # 构建粒子群
        for i in range(self.n):
            self._swarm.append(particle(self.fun[i],self.x[i],self.v[i]))
        self._p_opt = min(self._swarm,key=lambda p:p.y_opt)
        
    _intertiaWeightMethod = 'fixed','linear','random',
    def set_intertia_weight(self,method,*args):
        # 固定权重
        if method == PSO._intertiaWeightMethod[0]:
            self._intertiaWeight['MethodID'] = 0
            if args:
                self._intertiaWeight['args'] = args[0],
            else:
                self._intertiaWeight['args'] = 0.5,
        # 线性变化权重
        elif method == PSO._intertiaWeightMethod[1]:
            self._intertiaWeight['MethodID'] = 1
            if args:
                self._intertiaWeight['args'] = args[0],args[1] # (初始值，最终值)
            else:
                self._intertiaWeight['args'] = [0.9,0.4]
        # 随机权重
        elif method == PSO._intertiaWeightMethod[2]:
            self._intertiaWeight['MethodID'] = 2
            if args:
                self._intertiaWeight['args'] = args[0],args[1] # (下界，幅值)
            else:
                self._intertiaWeight['args'] = [0.5,0.5]
        else:
            raise ValueError("method should be in %s"%PSO._intertiaWeightMethod)
    def _get_intertia_weight(self):
        """will be called by self.iter and returns the current intertia weight."""
        if self._intertiaWeight['MethodID'] == 0:
            return self._intertiaWeight['args'][0]
        elif self._intertiaWeight['MethodID'] == 1:
            return self._intertiaWeight['args'][0] + (self._intertiaWeight['args'][1]-self._intertiaWeight['args'][0])*self.niter/self.iter_max
        elif self._intertiaWeight['MethodID'] == 2:
            return self._intertiaWeight['args'][0].random()*self._intertiaWeight['args']

    _learningFactorMethod = 'fixed','linear'
    def set_learning_factor(self,method,*args):
        if method == PSO._learningFactorMethod[0]:
            self._learningFactor['MethodID'] = 0
            if len(args) == 0:
                self._learningFactor['c1'] = 2,
                self._learningFactor['c2'] = 2,
            elif len(args) == 1:
                self._learningFactor['c1'] = args[0],
                self._learningFactor['c2'] = args[0],
            else:
                self._learningFactor['c1'] = args[0],
                self._learningFactor['c2'] = args[1],
        # 时变
        if method == PSO._learningFactorMethod[1]:
            self._learningFactor['MethodID'] = 1
            if len(args) == 0:
                self._learningFactor['c1'] = 2.5,0.5
                self._learningFactor['c2'] = 0.5,2.5
            elif len(args) == 2:
                self._learningFactor['c1'] = max(args),min(args)
                self._learningFactor['c2'] = self.c1[1],self.c1[0]
            else:
                self._learningFactor['c1'] = args[0],args[1]
                self._learningFactor['c2'] = args[2],args[3]
    def _get_learning_factor(self):
        sc1 = self._learningFactor['c1']
        sc2 = self._learningFactor['c2']
        if self._learningFactor['MethodID'] == 0:
            return sc1[0],sc2[0]
        elif self._learningFactor['MethodID'] == 1:
            c1 = sc1[0] + (sc1[1]-sc1[0])*self.niter/self.iter_max
            c2 = sc2[0] + (sc2[1]-sc2[0])*self.niter/self.iter_max
            return c1,c2


    
    def __iter__(self):
        """I'm the iterator!"""
        self.get_ready()
        return self
    def __next__(self):
        """Use the iterator to do the iteration."""
        if not self.niter < self._iter_max:
            self._msg = "iter times exceed iter_max"
            raise StopIteration(self._msg)
        elif max([i.y_opt-self._p_opt.y_opt for i in self._swarm]) < self.tol:
            self._msg = "all particles achieve the same value"            
            raise StopIteration(self._msg)
        else:
            self.iter_once()
    
    def iter_once(self):
        """Do iteration once."""
        w = self._get_intertia_weight()
        c1,c2 = self._get_learning_factor()
        # do the iteration
        for p in self._swarm:
            p.set_coeff(w=w,c1=c1,c2=c2)
            p.iter(self._p_opt)
        # find the new best particle
        self._p_opt = min(self._swarm,key=lambda p:p.y_opt)
        # count
        self._niter += 1
        if self._fun2:
            self._fun2(self._p_opt.x_opt,self._p_opt.y_opt,self.niter)

    def solve(self,full_output=False):
        rpt = {}
        for i in self:
            if full_output:
                print(self.niter,self._p_opt.y_opt)
        rpt['x'] = self._p_opt.x_opt
        rpt['fun'] = self._p_opt.y_opt
        rpt['msg'] = self._msg or 'success'
        rpt['niter'] = self.niter
        rpt['nparticle'] = self.n
        return OptimizeResult(rpt)
            
        
if __name__ == '__main__':
    from pylab import *
    from scipy.optimize import *
    seed(0)
    ps = PSO(rosen,random([40,8]),iter_max=1000)
    ps.set_intertia_weight('fixed')
    ps.set_learning_factor('fixed')

