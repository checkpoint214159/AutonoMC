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
from mineclip import MineCLIP

class Agent(nn.Module):
    """
    A minecraft agent built on the concepts of active inference.
    """

    def __init__(self, clip_config, num_policies):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = clip_config.pop('ckpt_path', None)

        # print('cfg', cfg)
        # print('device', device)
        self.encoder = MineCLIP(**clip_config).to(device)
        self.encoder.load_ckpt(path, strict=True)
        print("Successfully loaded MineCLIP.")

        self.preference_net = 1
        self.interpreter = 1
        self.transition_model = 1
        self.likelihood_model = 1 # P( o | s ). The question is, how do we learn such a model?

        self.num_policies = num_policies
        self.state_dicts = {}

        self.curr_timestep = 0

    def create_target_state(self, target):
        """
        Encode a target observation into latent observation into state. This is the end goal state the
        agent is trying to achieve. This end goal state technically shifts which each timestep, since although the 
        target observation is the same, how it interprets that target observation should change, as it itself changes.
        """


    def forward(self,
                observations,
                target,
            ):
        """
        Order of execution:
            1. Encode current observation into a latent observation.
            2. Calculate model surprise with respect to the current latent observation
            3. Minimize VFE
            4. Sample the several policies from the new Q model.
            5. Within each policy, calculate EFE.
            6. Update distribution over policies
            7. Repeat 4-6 until convergence.
            8. Sample a policy from the final distribution over policies
            9. Take an action according to this policy
            10. Repeat 1 - 9 for infinity lol or until you run out of iterations 
        """
        # 1 - 3:
        Q = self.perception(observations)

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
    
    def kl_divergence(self, mean1, mean2):
        """
        Calculate KL divergence given two univariate gaussians, according to 
        https://math.stackexchange.com/questions/2888353/how-to-analytically-compute-kl-divergence-of-two-gaussian-distributions
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2).sum()  # what even. idk bro, if gaussian is isotropic multivariate, its just this?
    
    def planning(
            self,
            policy,
        )

    def perception(
            self,
            observations
        ):
        """
        Creates an new model which has minimized VFE for the current observation.

        We can then sample from this model several policies to further refine them.

        1. Obtain current observation, and encode it into some latent observation state.
        2. Calculate VFE of this latent observation, with prior state at this timestep.
        3. Update model to minimize this VFE.
        4. Subsequently, 
        4. Repeat 2-3 until convergence.
        """

        latent_observation = self.encoder.forward_image_features(observations)  # observations must be 5d. tensor.
        q_s_t = self.likelihood_model(latent_observation)
        t = self.curr_timestep

        # If we are not at the zeroth timestep, find model which minimizes VFE.
        if t != 0:
            # First round of VFE with p_s_t calculated at the previous timestep, and update the model.
            p_s_t = self.state_dict[t - 1]
            self.kl_divergence(p_s_t, q_s_t)

            # Then, for subsequent rounds of inference sample some random timestep, and do VFH at that timestep.
            # for now, lets not write this code.
            # converged = False
            # while not converged:
            #     random_timestep = horizon


        # if we are at the zeroth timestep, there is no history to test our perceptions against. moving on!
        else:
            pass

 
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

conver
while not converged:
    random_timestep = generate_random_timestep()
    generate_state_dict(random_timestep)  # --> Q(s_t | s_t_m1, a)
    
    
    p_s_t, p_s_t_1 = self.state_dict[random_timestep], self.state_dict[random_timestep + 1]
    difference = kl_divergence(p_s_t, p_s_t_1)
    # do something, backprop something

    



















