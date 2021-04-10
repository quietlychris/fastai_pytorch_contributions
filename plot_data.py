#!/usr/bin/python3

# pip3 install polars numpy matplotlib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# import data using Polars, can be wi
fastai_df = pl.read_csv("fastai_contributions_15mo.csv",skip_rows=1)
fastai = fastai_df.to_numpy() # Convert from polars Dataframe to numpy array for plotting

pytorch_df = pl.read_csv("pytorch_contributions_15mo.csv",skip_rows=1)
pytorch = pytorch_df.to_numpy()

# Create primary figure with two sub-plots
fig, ax = plt.subplots(1,2)
fig.suptitle("Contribution Structures of PyTorch and FastAI (Top 25)")
# fig.set_titlepad(1.0)

scaling_factor = 1e-3
# PyTorch Data
contributors = pytorch[:,0]
adds = pytorch[:,1] * scaling_factor
subs = pytorch[:,2] * -1 * scaling_factor
ax[0].bar(range(0,len(adds)), adds, color='green')
ax[0].bar(range(0,len(subs)), subs, color='red')
ax[0].set_xlabel("Contributor #")
ax[0].set_ylabel("Additions and Subtractions [kLoC]")
ax[0].set_title("PyTorch")

# FastAI Data
contributors = fastai[:,0]
adds = fastai[:,1] * scaling_factor
subs = fastai[:,2] * -1 * scaling_factor
ax[1].bar(range(0,len(adds)), adds, color='green')
ax[1].bar(range(0,len(subs)), subs, color='red')
ax[1].set_title("FastAI")
ax[1].set_xlabel("Contributor #")


# plt.show()
plt.savefig('pytorch_fastai_contribution_structures.png',bbox_inches='tight')

