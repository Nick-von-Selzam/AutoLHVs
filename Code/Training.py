# Packages
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, random
import optax as optax
from jax.lax import scan, dynamic_slice, while_loop, fori_loop
from functools import partial


# Construct loss function from LHV and QM measurement rules and a distance measure for a fixed state
def loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state):
    
    PHV_up = vmap(LHV_measurement_rule, in_axes=(None, 0)) # vectorize over cloud-dimension
    def PHV(measures, cloud): # Average probability for the associated joint outcome given the hidden states
        p_ups = vmap(PHV_up, in_axes=(0, 1))(measures, cloud) # vectorize over particles
        return jnp.mean(jnp.prod(p_ups, axis=0)) # product over particles, mean over cloud

    def deviation(measures, cloud): # Deviation between probabilities predicted by LHV and QM
        phv = PHV(measures, cloud)
        pqm = QM_measurement_rule(measures, quantum_state)
        return distance_measure(pqm, phv)

    def Loss(cloud, batched_measures): # Loss = mean deviation over batch of combinations of measurements
        return jnp.mean(vmap(deviation, in_axes=(0, None))(batched_measures, cloud))
    
    return Loss


# Test loss on batch of measurement directions for a given distance measure, measurement rules and a sampling function
@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def test_loss(key, hidden_states, quantum_state, N_measures, LHV_measurement_rule, QM_measurement_rule, distance_measure, sample):

    Loss = loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state)

    N = hidden_states.shape[1]
    batched_measures = sample(key, N_measures, N)
    loss = Loss(hidden_states, batched_measures)
    
    return loss

# vectorize over key, hidden_states and quantum_state
test_loss_v = vmap(test_loss, in_axes=(*3*[0], *5*[None]))


# Train without diffusion steps
@partial(jit, static_argnums=tuple(range(3, 11)))
def autoLHV(key, cloud, quantum_state, 
           N_measures, N_measures_test, N_steps, 
           LHV_measurement_rule, QM_measurement_rule, distance_measure, sample, optimizer):

    '''
        Optimizes the hidden-variable distribution such that the measurement statistics of a given quantum state are reproduced as well as possible.

        Inputs:

            key = the initial key for the pseudo random number generator
        
            Data:
                cloud = (size of hidden-variable cloud, number of particles, hidden-variable dimension per particle)-tensor of initial hidden variables
                                the LHV_measurement_rule recieves a single hidden variable as input
                quantum_state = variable specifying the quantum state 
                                (e.g. a density matrix, correlation matrix, or one or more parameters specifying the state from some fixed family of states)
                                 the QM_measurement_rule function recieves this as input along with the measurement options
        
            Hyper parameters:
                N_measures = batch size 
                           = number of combinations of measurement options (POVM elements) the loss function is averaged over per gradient descent step
                N_measures_test = batch size for test loss evaluation (once, in the end)
                N_steps = total number of gradient descent steps
                                 
            Functions:
                LHV_measurement_rule: (measurement option, hidden-variable vector) -> probability for the associated measurement outcome
                QM_measurement_rule: (measurement option for each particle, quantum state) -> probability for the associated joint outcome
                distance_measure: (p_qm, p_lhv) -> some non-negative number quanitfying their deviation
                sample: (key, N_measures, N_spins) -> batch of measurement options, format needs to be compatible with LHV/QM_measurement_rule
                optimizer: An optax optimizer such as "Adam" with a learning rate (schedule)

        Outputs:
            cloud = the optimized hidden-variable cloud
            loss_values = array of loss value progression during training
            test_loss = loss for final cloud evaluated on N_measures_test many measurement combinations
    '''

    N = cloud.shape[1]
    Loss = loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state) # Construct the loss function
    opt_state = optimizer.init(cloud) # The initial parameters are the passed initial hidden states

    train_data = [key, opt_state, cloud] # These are updated at every training step

    def gradient_step(train_data, _):
        key, opt_state, cloud = train_data
        key, key_SGD = random.split(key)
        # Measurement combinations
        batched_measures = sample(key_SGD, N_measures, N)
        # SGD
        loss, grads = value_and_grad(Loss, argnums=0)(cloud, batched_measures)
        updates, opt_state = optimizer.update(grads, opt_state, cloud)
        cloud = optax.apply_updates(cloud, updates)
        return [key, opt_state, cloud], loss

    [key, opt_state, cloud], loss_values = scan(gradient_step, train_data, None, length=N_steps) # Training loop

    # Test loss
    key, key_test = random.split(key)
    batched_measures = sample(key_test, N_measures_test, N)
    test_loss = Loss(cloud, batched_measures)

    return cloud, loss_values, test_loss

# vectorize over key, hidden_states and quantum_state
autoLHV_v = vmap(autoLHV, in_axes=(*3*[0], *9*[None]))


# Train with diffusion steps
@partial(jit, static_argnums=(3, 4, 5, 6, 10, 11, 12, 13))
def autoLHV_noise(key, hidden_states, quantum_state, 
                N_measures, N_measures_test, N_steps, N_steps_noise, N_steps_finetune, noise, 
                LHV_measurement_rule, QM_measurement_rule, distance_measure, sample, optimizer):
    
    '''
        Optimizes the hidden state distribution such that the measurement statistics of a given quantum state are reproduced as well as possible.

        Inputs:

            key = the initial key for the pseudo random number generator
        
            Data:
                hidden_states = (size of hidden state cloud, number of spins, hidden state dimension per spin)-tensor of initial hidden states
                                the LHV_measurement_rule recieves a single hidden state as input
                quantum_state = variable specifying the quantum state 
                                (e.g. a correlation matrix, or one or more parameters specifying the state from some fixed family of states)
                                 the QM_measurement_rule function recieves this as input along with the measurement options
        
            Integer hyper parameters:
                N_measures = batch size 
                           = number of combinations of measurement options the loss function is averaged over per gradient descent step
                N_measures_test = batch size for test loss evaluation (once, in the end)
                N_steps = total number of gradient descent steps
                N_steps_fine_tune: for the last N_steps_fine_tune gradient descent steps a 10x smaller learning_rate is used
                N_steps_noise = number of gradient descent steps followed by a diffusion step, needs to be smaller or equal to N_steps

            Float hyper parameters:
                noise = strength of the gaussian noise added in the diffusion steps
                                 
            Functions:
                LHV_measurement_rule: (measurement option, hidden state vector) -> probability for up
                QM_measurement_rule: (measurement option for each spin, quantum state) -> probability for up along all directions
                distance_measure: (p_qm(up,...,up), p_lhv(up,...up)) -> some non-negative number quanitfying their deviation
                sample: (key, N_measures, N_spins) -> batch of measurement options, format needs to be compatible with LHV/QM_measurement_rule
                optimizer: An optax optimizer such as "Adam" with a learning rate (schedule)

        Outputs:
            hidden_states = the optimized hidden state cloud
            loss_values = array of loss value progression during training
            test_loss = loss for final hidden_states evaluated on N_measures_test many measurement combinations
    '''

    N_spins = hidden_states.shape[1]
    Loss = loss_from_rules(LHV_measurement_rule, QM_measurement_rule, distance_measure, quantum_state) # Construct the loss function
    opt_state = optimizer.init(hidden_states) # The initial parameters are the initial hidden states

    train_data = [key, opt_state, hidden_states] # These are updated at every training step
    
    # Add diffusion step after gradient descent step for first N_steps_noise steps
    noise_schedule = jnp.concatenate([noise*jnp.ones(N_steps_noise), jnp.zeros(N_steps-N_steps_noise)])

    def gradient_step(train_data, noise):
        key, opt_state, hidden_states = train_data
        key, key_SGD, key_Diffusion = random.split(key, (3,))
        # Measurement combinations
        batched_measures = sample(key_SGD, N_measures, N_spins)
        # SGD
        loss, grads = value_and_grad(Loss, argnums=0)(hidden_states, batched_measures)
        updates, opt_state = optimizer.update(grads, opt_state, hidden_states)
        hidden_states = optax.apply_updates(hidden_states, updates)
        # Diffusion
        hidden_states += noise*random.normal(key_Diffusion, shape=hidden_states.shape)
        return [key, opt_state, hidden_states], loss
    
    [key, opt_state, hidden_states], loss_values = scan(gradient_step, train_data, noise_schedule) # Training loop

    # Test loss
    key, key_test = random.split(key)
    batched_measures = sample(key_test, N_measures_test, N_spins)
    test_loss = Loss(hidden_states, batched_measures)

    return hidden_states, loss_values, test_loss

# vectorize over key, hidden_states and quantum_state
autoLHV_noise_v = vmap(autoLHV_noise, in_axes=(*3*[0], *11*[None]))
