"""
Author: Miles Winter
Desc: Use PyMC to construct a MCMC that will estimate
      the parameters of a broken-power-law distribution
      using a regression model. 
"""

from __future__ import division
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
import scipy.stats as stats
import corner
plt.style.use('ggplot')
np.random.seed(1234)

#set up optional arguments
p = argparse.ArgumentParser(description="optional arguments")
p.add_argument("-i", "--iterations", dest="iterations", type=int, default=150000, 
               help="Number of MCMC sample")
p.add_argument("-b", "--burn", dest="burn", type=int, default=100000, 
               help="Number of burn-in samples")
p.add_argument("-t", "--thin", dest="thin", type=int, default=1, 
               help="thin by only retaining every kth sample, where k is an integer")
p.add_argument("-s", "--sigma", dest="sigma", type=float, default=0.5, 
               help="Width of noise distribution")
p.add_argument("-k", "--k_val", dest="k_val", type=float, default=2., 
               help="Value of normalization const in log10 space")
p.add_argument("-xb", "--xb_val", dest="xb_val", type=float, default=0., 
               help="Value of break in log10 space")
p.add_argument("-a1", "--a1_val", dest="a1_val", type=float, default=0.5, 
               help="Value of low end slope in log10 space")
p.add_argument("-a2", "--a2_val", dest="a2_val", type=float, default=2.5, 
               help="Value of high end slope in log10 space")
p.add_argument("-p", "--par_vals", dest="par_vals", nargs=4, default=None, type=float,
               help="space-separated list of parameter values, i.e. -p k xb a1 a2")
args = p.parse_args()

pars = []
if args.par_vals:
    pars = args.par_vals
else:
    pars = [args.k_val, args.xb_val, args.a1_val, args.a2_val]


def broken_power_law(x, k, xb, a1, a2):
    """Broken power law in log10 space """
    dN = -a1*x[x<=xb]+k
    dN = np.append(dN, -a2*(x[x>xb]-xb)-a1*xb+k)
    return dN

#Define observed data
Logx = np.linspace(-2,2,41) 
LogF = broken_power_law(Logx, *pars)

#add gaussian noise
LogF_obs = LogF + np.random.normal(0,args.sigma,size=len(LogF))

# define priors
k = pymc.Normal('k', mu=0., tau=0.01)
xb = pymc.Normal('xb', mu=0., tau=0.01)
a1 = pymc.Normal('a1', mu=0., tau=0.01)
a2 = pymc.Normal('a2', mu=0., tau=0.01)
tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

# define likelihood
@pymc.deterministic
def mu(x=Logx, k=k, xb=xb, a1=a1, a2=a2):
    dN = -a1*x[x<=xb]+k
    dN = np.append(dN, -a2*(x[x>xb]-xb)-a1*xb+k)
    return dN

y = pymc.Normal('LogF', mu=mu, tau=tau, value=LogF_obs, observed=True)

#Define Model
m = pymc.Model([k, xb, a1, a2, tau, Logx, y])

#Normal approx to get covariance matrix
N=pymc.NormApprox(m)
N.fit()
cov_mat = N.C[k, xb, a1, a2, tau]
np.savetxt('covariance_matrix.txt',cov_mat)

#MCMC to determine fit
mc = pymc.MCMC(m)
mc.sample(iter=args.iterations, burn=args.burn, thin=args.thin)

samples = np.array([k.trace(),xb.trace(),a1.trace(),a2.trace()]).T
tmp = corner.corner(samples[:,:], 
                    labels=[r'$k$',r'$xb$',r'$\alpha_1$',r'$\alpha_2$'], 
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, 
                    title_kwargs={"fontsize": 12})

tmp.gca().annotate("MCMC Summary and Parameter Covariance", 
                    xy=(0.5, 1.0), 
                    xycoords="figure fraction", 
                    xytext=(0, -5), 
                    textcoords="offset points", 
                    ha="center", 
                    va="top")

tmp.savefig('tri_plot.png')

ktr = k.trace()
xbtr = xb.trace()
a1tr = a1.trace()
a2tr = a2.trace()

kbar = k.stats()['mean']
xbbar = xb.stats()['mean']
a1bar = a1.stats()['mean']
a2bar = a2.stats()['mean']

trace_array = np.array([ktr,xbtr,a1tr,a2tr])
np.savetxt('trace_array.txt',trace_array)

data = pd.DataFrame(np.array([Logx, LogF_obs]).T, columns=['Logx', 'LogF'])
data.plot(x='Logx', y='LogF', kind='scatter', s=25);
plt.plot(Logx,broken_power_law(x=Logx, k=kbar, xb=xbbar, a1=a1bar, a2=a2bar))
plt.savefig('best_fit_BPL.png')
plt.show()

mc.write_csv("par_summary.csv", variables=["k", "xb", "a1", "a2"])

