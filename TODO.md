# TODO

0. Git good

## Step 1: Resolve policy-making first

Policy parameterizes transition "matrixes" (functions); under each policy, B is different (how states evolve will differ) so to reconcile perception, we must first reconcile how to "create" or at least sample policies. Presume there exists a space of policies, where we define each policy to be a parameterization of the self. This reshapes policies from 'a sequence of actions' to 'parameters of a generative model'! A natural hierarchy emerges, where this is the 'higher level model' that models the lower level perception processes (?).

In active inference, a policy is described as a sequence of actions. By the above paragraph, and if considering an action as a transition between two states in state space, this 'sequence of actions' is implicit within whatever the policy is. The question as to how we optimize such a policy generative / predictive network is via the expected free energy of each generated policy: for policies that better lower expected free energy, loss decreases. How this might work if our policies are continuous is a different story, so lets just think that we have some countable set of policies on hand for now.

Above this is the idea of 'preferences'; some underlying bias that guides what policies we choose. I.e, given some ideal observation we want to recieve (thus some ideal state we want to be in), and given our prior preferences, generate some amount of task-specific policies (state-transition parameters) to achieve the ideal observation. Now, the existing active inference literature sees the probability that an agent selects a policy as proportional to the inverse of their expected free energy, as in policies with lower EFEs are selected with greater probability. Perhaps instead of having a set of finite policies, we find some way to describe a distribution over policy space, then sample from this space to get policies to parameterize state transitions? Thats a whole lot of nested modelling lol.

### Iterative policy refinement?

From a conditioning on some ideal target state / encoded preferences given some task, predict a vector representing a certain policy. This vector parameterizes some generative model P subscript pi.