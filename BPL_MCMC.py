from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
import scipy.stats as stats
import corner
plt.style.use('ggplot')
np.random.seed(1234)


def broken_power_law(k, Lb, a1, a2, L):
    """Broken power law in log10 space """
    dN = -a1*L[L<=Lb]+k
    dN = np.append(dN, -a2*(L[L>Lb]-Lb)-a1*Lb+k)
    return dN

#Define observed data
LogL = np.linspace(-2,2,41)
LF = broken_power_law(k=2.,Lb=0.,a1=0.5,a2=2.5,L=LogL) 

#add gaussian noise
LF_obs = LF + np.random.normal(0,0.3,size=len(LF))

# define priors
k = pymc.Normal('k', mu=0., tau=0.01)
Lb = pymc.Normal('Lb', mu=0., tau=0.01)
a1 = pymc.Normal('a1', mu=0., tau=0.01)
a2 = pymc.Normal('a2', mu=0., tau=0.01)
tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

# define likelihood
@pymc.deterministic
def mu(k=k, Lb=Lb, a1=a1, a2=a2, L=LogL):
    dN = -a1*L[L<=Lb]+k
    dN = np.append(dN, -a2*(L[L>Lb]-Lb)-a1*Lb+k)
    return dN

y = pymc.Normal('LF', mu=mu, tau=tau, value=LF_obs, observed=True)

#Define Model
m = pymc.Model([k, Lb, a1, a2, tau, LogL, y])

#Normal approx to get covariance matrix
N=pymc.NormApprox(m)
N.fit()
cov_mat = N.C[k, Lb, a1, a2, tau]
np.savetxt('covariance_matrix.txt',cov_mat)

#MCMC to determine fit
mc = pymc.MCMC(m)
mc.sample(iter=500000, burn=450000)

samples = np.array([k.trace(),Lb.trace(),a1.trace(),a2.trace()]).T
tmp = corner.corner(samples[:,:], 
                    labels=[r'$k$',r'$Lb$',r'$\alpha_1$',r'$\alpha_2$'], 
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

tmp.savefig('tri_plot.pdf')

ktr = k.trace()
Lbtr = Lb.trace()
a1tr = a1.trace()
a2tr = a2.trace()

kbar = k.stats()['mean']
Lbbar = Lb.stats()['mean']
a1bar = a1.stats()['mean']
a2bar = a2.stats()['mean']

trace_array = np.array([ktr,Lbtr,a1tr,a2tr])
np.savetxt('trace_array.txt',trace_array)

data = pd.DataFrame(np.array([LogL, LF_obs]).T, columns=['LogL', 'LF'])
data.plot(x='LogL', y='LF', kind='scatter', s=25);
plt.plot(LogL,broken_power_law(k=kbar, Lb=Lbbar, a1=a1bar, a2=a2bar, L=LogL))
plt.savefig('best_fit_BPL.png')
plt.show()

mc.write_csv("par_summary.csv", variables=["k", "Lb", "a1", "a2"])

