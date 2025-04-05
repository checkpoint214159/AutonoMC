"""
This module is meant to encapsulate the entire agent and contain all its components.
This includes (as of now):
1. Preference module (the guy that generates a distribution over policies)
2. Interpretation/Likelihood module (the guy that deterministically / stochastically maps states
    to observations, or vice versa)
3. Transition module (the guy that describes timewise state transitions)

Its forward method controls the flow of everything.
"""

import torch
from torch import nn

class Agent(nn.Module):
    """
    A minecraft agent built on the concepts of active inference.
    """

    def __init__(self, num_policies):

        self.preference_net = 1
        self.interpreter = 1
        self.transition_model = 1
        self.likelihood_model = 1 # P( o | s ). The question is, how do we learn such a model?

        self.num_policies = num_policies
        self.state_history = []


    def forward(self,
                observations,
                target,
            ):
        """
        Order of execution:
            1. Encode current observation into a latent observation.
            2. Calculate model surprise with respect to the current latent observation
            3. Generate Q model which minimizes variational free energy of the current state
            4. Sample the several policies from the new Q model.
            5. Within each policy, calculate EFE.
            6. Update distribution over policies
            7. Repeat 4-6 until convergence.
            8. Sample a policy from the final distribution over policies
            9. Take an action according to this policy
            10. Repeat 1 - 9 for infinity lol or until you run out of iterations 
        """
        # 1 - 3:
        Q = self.forward_vfe(observations)

        # 4 - 7:
        converged = False
        while not converged:
            for i in range(self.num_policies):
                pi = Q.sample  # some sample method
                
                # minimize vfe

        # 8 - 10:
        policy = Q.sample
        action = policy.sample()

        return action

    def forward_vfe(
            self,
            observations
        ):
        """
        Creates an new model which has minimized VFE for the current observation.

        We can then sample from this model several policies to further refine it
        to minimize EFE.

        Here is how it is done in the current literature (if I am not mistaken)
        1. Obtain current observation.
        2. Calculate VFE with prior states.
        3. Update prior states based on some error.
        4. Repeat 2-3 until convergence.

        Now, we make a changes. Because we are dealing with inputs of insane complexity themselves, we must
        encode the current observations into a 'latent' observation. This should be distinct from the idea of 'latent states', since
        a 'latent observation' would be some compressed interpretation of what you are observing, whilst the latent state would (should)
        encode some larger idea of the world BEYOND that which is observable. The final process will be as such:
        1. Obtain current observation, and encode it into some latent observation state.
        2. Calculate VFE of this latent observation, with prior states.
        3. Update prior states based on some error.
        4. Repeat 2-3 until convergence.
        """

        # for now, assume deterministing tings 
        latent_observation = self.encode_obs(observations)
        mean_latent = self.likelihood_model(latent_observation)  # as of now, a (dim, 1) shape vector representing 1 mean per dimension.
        # in the future could have more than one mean to denote the sum of multiple gaussians? idk
        t = self.curr_timestep

        # If we are at the start timestep, just predict future priors.
        if t == 0:
            pass
        # else if we are at the first timestep and beyond, we can start minimizing VFE.
        else:
            # if we are at the first timestep, the very first round of VFE updating only consist of a message passed from latent observations.
            # this is because the model has not been updated yet since the last timestep, since no planning over policies was done yet.
            # in subsequent rounds of VFE updating, since the model has been updated and the state transitions for the previous and future timesteps
            # have been updated, these factors can be used in updating.
            if t == 1:
                pass
            # if we are in subsequent timesteps, the model has already been updated from the previous round of VFE planning.
            # thus, proceed with normal VFE minimization as expected. (3 factors if curr observation is present, 2 if we are doing it
            # in the past or future)
            else:
                pass
        
        # error = 0.5 * log(transition model * s - 1) + log(transition model * s at t+ 1) + log likelihood * curr_obs - curr state
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
print(y)   # --> p(x)

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x, y)
plt.savefig('p(x).png')

x, y = np.mgrid[-1:1:.01, -1:1:.05]
print('x and y:')
print(x, x.shape)
print(y, y.shape)
pos = np.dstack((x, y))
print('pos', pos, pos.shape)
rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 2.0]])  # mean, covar
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, rv.pdf(pos))
print('rv.pdf pos', rv.pdf(pos))
plt.savefig('waht3.png')
