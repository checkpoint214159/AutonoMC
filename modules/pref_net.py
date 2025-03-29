import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class PrefNet(nn.Module):
    """
    A module that produces some parameters of a policy distribution.
    The actual neural network is intended to represent how, given some target
    end observation, we come up with various policies to achieve that goal. 

    PrefNet: o_target -> policy_dist_params

    From there, find the expected free energy of each policy, which gives us an
    idea of what policies might do better. This allows us to form a posterior over policies,
    from which we find some kind of divergence / difference with prior policy distribution,
    updating it. 
    
    Iteratively do this until the divergence / difference is small enough,
    then we backprop for PrefNet?

    Right now we presume that the elements of the latent state are independent, so I don't need to 
    spam a covariance matrix that wld annihilate me.
    """
    def __init__(
        self,
        embed_dim,
        in_dim,
        num_normals,
    ):  
        super().__init__()

        self.num_normals = num_normals

        self.mean_networks = [nn.Linear(in_dim, embed_dim) for i in range(num_normals)]
        self.var_networks = [nn.Linear(in_dim, embed_dim) for i in range(num_normals)]
        self.softmax = nn.Softmax(dim=0)
        # self.means = Parameter(torch.zeros(num_normals, embed_dim))
        # self.vars = Parameter(torch.zeros(num_normals, embed_dim))

        # torch.nn.init.xavier_uniform(self.means)
        # torch.nn.init.xavier_uniform(self.vars)

    def forward(
        self,
        batched_input,
        num_policies=2,
    ):
        
        # for now screw it and throw inputs to the seperate mean networks in a for loop
        # there is probably a much faster way to do it
        mean = []
        vars = []
        for i in range(self.num_normals):
            m = self.mean_networks[i](batched_input)
            v = self.var_networks[i](batched_input)
            mean.append(m)
            vars.append(v)

        mean = self.softmax(torch.cat(mean, dim=1))
        vars = self.softmax(torch.cat(vars, dim=1))
        # print(mean.shape, 'meanshape')
        # print(vars.shape, 'varsshape')

        with torch.no_grad():
            normal = Normal(mean, vars)
            policy_params = normal.sample((num_policies,))  # cannot be iterable

            # print(policy_params)
            print(policy_params.shape)

prefnet = PrefNet(512, 128, 3)
input = torch.randn((3, 1, 128))
prefnet(input, num_policies=3)
# print(dir(prefnet))