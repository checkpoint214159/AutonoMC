
"""
Basic perciever:

Base idea; for any timestep t,  there exists a prior over state transitions
and likelihood of observations under given states. 

Initially, it has some beliefs over what the state could be at this timestep
and what observations it could recieve. p(s_t), p(o_t|s_t). Should this be random as the paper said?

Based on previous and future state s_t-1, s_t+1, and observation o_t, 
predict the current state s_t. 

Rinse and repeat across all equality nodes.
"""

"""
How to encode likelihood / transition matrices?

Transition matrix
Could use multiple multivariate gaussians; predict mean, var for each one and take their normalized sum
This has the effect of 'predicting modes'? Per state factor, parameterize a few gaussians = transition matrix

Likelihood matrix
Likewise; just have it be the same dimensionality as the state one, then later decode it into actual observations to compare with in 
pixel / sensory space if need be (or compare in latent)
"""

# ------ START ------

# init model prior, posteriors, etc.
model = Model()
q = model.init_posteriors()  # all are q(s_t), randomly generated? size is (T, 2) where 2 is for mean, variance. Gaussian assumption ofc.

policy = model.habits
o_t = env.take_observation()

while not done:
    t = random.sample(T)  # assert t >= 1.
    p_t = model.get_p(t, policy)
    likelihood_t = model.get_likelihood(t, policy)  # (T, 2) means, (T, TODO: something) var
    transition_t = model.get_transition(t, policy)  # (T, 2) means, (T, TODO: something) var

    # TODO: There is no t = 0. Should t = 1, message_tm1 is just q[1]. Otherwise, message_tm1 is DERIVED from q[t-1] times transition.
    q_tm1, q_tp1 = q[t-1], q[t+1]
    
    # TODO: The following is for the imaginary matrix case. in reality it will be inverting gaussians and calculating from there
    message_tm1 = q_tm1 * transition_t
    message_tp1 = q_tp1 * transition_t.transpose  # transition_t : rows are s_t, cols are s_t-1
    likelihood_factor = o_t * likelihood_t.transpose  # likelihood_t: rows are o_t, cols are s_t

    q_t = message_tm1 + message_tp1 + likelihood_factor

    vfe = model.calc_vfe(q_t, p_t)

    # insert some variational message passing algo here
    vfe.backwards()

# ------ END ------

"""
What about habits? The literature defines an action to be a transition between states, and a policy to be a sequence of actions.
However, a transition between states is precisely what the state transition matrix expresses the probability of, so in this
way, what 'a' is, is probabilistic (according to the vector/distribution that describes the probability of a state factor 
transitioning to another state factor). Does this mean we can describe a policy as parameterizing the transition matrices? (i.e parameterizing
the parameters of the transition matrix)?

To 'select' an action, sample from the transition matrix sequentially, i.e at t=2, sample latent state s_2 from the calculated
p(s_2) ( or q(s_2) ?? ) distribution. Then, maybe have some model f(s_t, s_t+1) predict action a_t needed to transition? This will of course
be in the latent space ( maybe lets call it a latent action space ), where the factors ( embedding dimension ) are self-learned.
Then use another model to predict the actual physical action(s) represented by a_t.

Compared to discrete and hand-crafted states, there is no guarantee the actor can take the appropriate real physical action in the environment 
that would actually correspond to the state described in this probabilistic embedding space, which is a major downside. Also, how to 'relate' and
backprop the actions to observations? We could have a perfectly fine generative model, but if the actions taken are nonsense it might not be
able to actively learn at all.
"""

# ------ START ------

# init model prior, posteriors, etc.
model = Model()
q = model.init_posteriors()  # all are q(s_t), randomly generated? size is (T, 2) where 2 is for mean, variance. Gaussian assumption ofc.

policy = model.habits
o_t = env.take_observation()

while not done:
    t = random.sample(T)  # assert t >= 1.
    p_t = model.get_p(t, policy)
    likelihood_t = model.get_likelihood(t, policy)  # (T, 2) means, (T, TODO: something) var
    transition_t = model.get_transition(t, policy)  # (T, 2) means, (T, TODO: something) var

    # TODO: There is no t = 0. Should t = 1, message_tm1 is just q[1]. Otherwise, message_tm1 is DERIVED from q[t-1] times transition.
    q_tm1, q_tp1 = q[t-1], q[t+1]
    
    # TODO: The following is for the imaginary matrix case. in reality it will be inverting gaussians and calculating from there
    message_tm1 = q_tm1 * transition_t
    message_tp1 = q_tp1 * transition_t.transpose  # transition_t : rows are s_t, cols are s_t-1
    likelihood_factor = o_t * likelihood_t.transpose  # likelihood_t: rows are o_t, cols are s_t. This is kinda sudo since observation might not be a one-hot vector.

    q_t = message_tm1 + message_tp1 + likelihood_factor

    calc_vfe = model.calc_vfe(q_t, p_t)

    # insert some variational message passing algo here
    vfe.backwards()

# once done, sample / get actions.

for t in range(T):
    p_t = model.get_prior_at(t)
    p_tp1 = model.get_prior_at(t + 1)
    transition_t = model.get_transition(t, policy)  # (T, 2) means, (T, TODO: something) var

    # can difference in the state space really translate to good actions?
    a_t = p_tp1 - p_t  
    action_seq_at_t = model.interpret_action(a_t)

# ------ END ------

"""
How do we encode preferences?

a) Preferences are conditional on the current state
    - This would mean that the model predicts its own future; it creates preferences for policies / observations based on 
    its current state. That also means current state must somehow encode within it a future ideal state / observation ?
    Trippy and very philosophy pilled.

b) Preferences are conditional on some target   
    - Given a target, model generates preferences. This resolves the aforementioned trippiness of states encoding their own
    future (or maybe thats not that trippy), but means that this target is beyond the model's world, which means literal
    injection from a higher order being (self-motivationg doesnt count, that should be in propositon a). ). Trippy and 
    very philosophy pilled again, but fits more squarely into the usual DL/ML pipeline and paradigm.
"""

obs_seq = model.predict_obs(preferences)  # embedding space, idea of actual obs?



    
    # pic = obs['rgb']
    # pic = pic.transpose((1, 2, 0))
    # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    
    