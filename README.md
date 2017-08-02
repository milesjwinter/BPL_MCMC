# Broken-Power-Law MCMC
Generate observation data from a broken power law, inject gaussian noise, and fit data using a MCMC regression model in PyMC. Then, make cool triangle plots with Corner. 

![BPL](sample_output/best_fit_BPL.png)


![tri_plot](sample_output/tri_plot.png)

## Usage
Running 
```bash
$ python BPL_MCMC.py
```
will generate,
 - par_summary.csv, containing the final parameter summaries
 - covariance_matrix.txt, containing an estiamte of the covariance matrix
 - trace_array.txt, containing the trace for each parameter
 - best_fit_BPL.png, plot of the best-fit line using expectation values for each parameter
 - tri_plot.pdf, a triangle plot of contours and posterior distributions made with Corner

The following optional arguments are available
- -i, number of MCMC samples: default=150000
- -b, number of burn-in samples: default=100000
- -t, thin by only retaining every kth sample, where k is an integer: default=1 (keep everything)
- -s, width of the noise distribution: default=0.5
- -k, value of normalization const in log10 space: default=2.
- -xb, x value of break in log10 space: default=0.
- -a1, value of low end slope in log10 space: default=0.5
- -a2, value of high end slope in log10 space: default=2.5
- -p, space-separated list of all 4 parameter values (instead of changing them individually), i.e. -p k xb a1 a2: default=None

Example: Run with 200000 MCMC samples, keeping every 10th
```bash
$ python BPL_MCMC.py -i 200000 -t 10
```


## Dependencies
PyMC, Corner, and Pandas can be installed with

```bash
$ pip install -r requirements.txt
```
[PyMC Documentation](http://pymc-devs.github.io/pymc/)

[Corner Documentation](http://corner.readthedocs.io/en/latest/)

