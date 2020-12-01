import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial

from cg_dynamics.compass_gait import Dynamics

class NeuralNet:

    def __init__(self, net_dims, args, opt_name, opt_kwargs={}):
        """Constructor for neural network class used to approximate CBF.
        Args:
            net_dims: Dimensions of layers in neural network.
            hyperparams: Hyper-parameters used to evaluate loss function.
            opt_name: Name of optimization algorithm to use.
            opt_kwargs: Named keyword arguments for optimizer.
        """

        self.net_dims = net_dims
        self.args = args
        self.dyn = Dynamics()
        self.opt_name = opt_name
        self._init_optimizer(opt_kwargs)
        self.dual_vars = self._init_dual_variables(verbose=True)
        self.dual_step_size = 0.05

        self.key = jrandom.PRNGKey(5433)

        # self.forward is a function that passes a batch of data through the NN
        self.forward = jax.vmap(self.forward_indiv, in_axes=(0, None))

    def __call__(self, x, params):
        """PyTorch-like calling of class-based neural network forward method."""

        return self.forward(x, params)

    @partial(jax.jit, static_argnums=0)
    def step(self, epoch, opt_state, dataset):
        """Do one step of optimization based on self.loss.
        Args:
            epoch: Currrent step of training.
            opt_state: Current state of network parameters.
            dataset: Training dataset.
        Returns:
            opt_state of NN parameters after this optimization step.
        """

        params = self.get_params(opt_state)
        grads = jax.grad(self.loss, argnums=0)(params, dataset)

        return self.opt_update(epoch, grads, opt_state)

    # @partial(jax.jit, static_argnums=0)
    def dual_step(self, params, dataset):

        _, _, diffs = self.loss_and_constraints(params, dataset)

        def dual_update(name):
            dv = self.dual_vars[f'λ_{name}']
            const = diffs[name]
            return jnn.relu(dv + self.dual_step_size * jnp.sum(const) / const.shape[0])

        self.dual_vars['λ_safe'] = dual_update('safe')
        self.dual_vars['λ_unsafe'] = dual_update('unsafe')
        self.dual_vars['λ_cnt'] = dual_update('cnt')
        self.dual_vars['λ_dis'] = dual_update('dis')

    @partial(jax.jit, static_argnums=0)
    def loss(self, params, dataset):
        loss, _, _ = self.loss_and_constraints(params, dataset)
        return loss

    @partial(jax.jit, static_argnums=0)
    def constraints(self, params, dataset):
        _, constraints, _ = self.loss_and_constraints(params, dataset)
        return constraints

    @partial(jax.jit, static_argnums=0)
    def loss_and_constraints(self, params, dataset):
        """Calculate loss for training neural network subject to CBF constraintsself.
        Args:
            dataset: Training dataset.
            params: Trainable parameters of the neural network.
        Returns:
            Value of the loss function.
        """

        def forward(x):
            return jax.vmap(self.forward_indiv, in_axes=(0, None))(x, params)

        def curr_cbf(x):
            return jax.vmap(self.cbf_term_indiv, in_axes=(0, None))(x, params)

        def forward_grad(x):
            return jax.vmap(self.model_grad_indiv, in_axes=(0, None))(x, params)

        def soft_constraint(vect):
            return jnp.sum(jnn.relu(vect))

        def hard_constraint_pct(vect):
            frac_incorrect = jnp.sum(jnp.heaviside(vect, 0)) / vect.shape[0]
            return (1.0 - frac_incorrect) * 100.0

        # Goal: enforce satisfaction of the hybrid CBF constraint 
        #       h(z) >= \gamma_safe     \forall cts and dis states
        x_safe = jnp.vstack((dataset['x_cts'], dataset['x_dis_plus']))
        x_safe_diff = self.args.gam_safe - forward(x_safe)
        safe_loss = soft_constraint(x_safe_diff)
        safe_const_pct = hard_constraint_pct(x_safe_diff)

        # Goal: enforce constraint on unsafe/boundary points
        #       h(x) <= -\gamma_unsafe      \forall boundary states
        # x_unsafe_diff = forward(dataset['x_unsafe']) + self.hyperparams['gamma_unsafe']
        x_unsafe_diff = forward(dataset['x_unsafe']) + self.args.gam_unsafe
        unsafe_loss = soft_constraint(x_unsafe_diff)
        unsafe_const_pct = hard_constraint_pct(x_unsafe_diff)

        # Goal: enforce the continuous state constraint with L_inf bound on actions
        # cnt_diff = self.hyperparams['gamma_cnt'] - curr_cbf(dataset['x_cts'])
        cnt_diff = self.args.gam_cnt - curr_cbf(dataset['x_cts'])
        continuous_loss = soft_constraint(cnt_diff)
        cnt_const_pct = hard_constraint_pct(cnt_diff)

        # Goal: enforce the discrete state constraint
        #       sup_{u_d \in U_d} h(f_d(z) + g_d(z)u_d) >= 0
        # disc_diff = self.hyperparams['gamma_dis'] - forward(dataset['x_dis_minus'])
        disc_diff = self.args.gam_dis - forward(dataset['x_dis_minus'])
        discrete_loss = soft_constraint(disc_diff)
        disc_const_pct = hard_constraint_pct(disc_diff)

        # Goal: make all derivatives small
        x_all = jnp.vstack((
            dataset['x_cts'], dataset['x_dis_minus'], dataset['x_dis_plus'], dataset['x_unsafe']
        ))
        dh_loss = jnp.sum(jnp.square(forward_grad(x_all)))

        # Goal: penalize large Lipschitz constants of the NN
        param_loss = jnp.sum(jnp.square(ravel_pytree(params)[0]))

        total_loss = (
            self.dual_vars['λ_safe'] * safe_loss +
            self.dual_vars['λ_unsafe'] * unsafe_loss +
            self.dual_vars['λ_cnt'] * continuous_loss +
            self.dual_vars['λ_dis'] * discrete_loss +
            self.dual_vars['λ_grad'] * dh_loss +
            self.dual_vars['λ_param'] * param_loss
        )

        all_const_pcts = {
            'safe': safe_const_pct,
            'unsafe': unsafe_const_pct,
            'cnt': cnt_const_pct,
            'disc': disc_const_pct,
        }

        all_diffs = {
            'safe': x_safe_diff,
            'unsafe': x_unsafe_diff,
            'cnt': cnt_diff,
            'dis': disc_diff
        }

        return total_loss, all_const_pcts, all_diffs

    def cbf_term_indiv(self, x, params):
        """Calculates the LHS of the CBF inequality term (without the sup)
                (dh/dx)f(x) + (dh/dx)g(x)u + alpha(h(x))
        Args:
            x: State input to neural network.
            params: Learnable parameters of neural network.
        Returns:
            LHS of control barrier function inequality.
        """

        def abs_l1_smooth(x, mu):
            return mu * jnp.log(jnp.cosh(x / mu))

        def l1_smoothing_indiv(x, mu):
            return jnp.sum(abs_l1_smooth(x, mu))

        alpha = lambda x: x
        dh = self.model_grad_indiv(x, params)

        term1 = jnp.dot(self.dyn.f(x), dh)
        term2 = jnp.linalg.norm(jnp.dot(dh.T, self.dyn.g(x)), ord=1)
        term3 = alpha(self.forward_indiv(x, params))

        return term1 + term2 + term3

    def model_grad_indiv(self, x, params):
        """Calculate gradient of the NN model h(x) WRT inputs x.
        Args:
            x: State input to neural network.
            params: Learnable parameters of neural network.
        Returns:
            Gradient dh/dx of model with respect to inputs.
        """

        return jax.grad(self.forward_indiv)(x, params)

    @staticmethod
    def forward_indiv(x: jnp.ndarray, params: list):
        """Forward pass through the neural network for a single instance.
        Args:
            x: Single instance to input into NN.
            params: Trainable parameters of neural network.
        Returns:
            Output from passing instance through neural network.
        """

        activations = x
        for weight, bias in params[:-1]:
            outputs = activations.dot(weight) + bias
            activations = jnp.tanh(outputs)

        final_weight, final_bias = params[-1]
        out = activations.dot(final_weight) + final_bias

        return jnp.squeeze(out)

    @staticmethod
    def random_layer_params(m: int, n: int, key):
        """Randomly intialize a weight matrix and bias vector.
        Args:
            m, n: Dimensions of weight matrix.
            key: Jax random number generator.
            scale: Scale of random initialization.
        Returns:
            weight: Randomly initialized weight matrix.
            bias: Randomly initialized bias vector.
        """

        w_key, b_key = jrandom.split(key)
        weight = jrandom.normal(w_key, (m, n))
        bias = jrandom.normal(b_key, (n, ))

        return weight, bias

    def init_params(self, verbose=False):
        """Intialize the parameters of the neural network with random values."""

        keys = jrandom.split(self.key, len(self.net_dims))
        dimensions = zip(self.net_dims[:-1], self.net_dims[1:], keys)
        params = [self.random_layer_params(m, n, k) for (m, n, k) in dimensions]

        if verbose is True:
            print(f'Param shapes are: {[(p1.shape, p2.shape) for (p1, p2) in params]}')

        return params

    def _init_dual_variables(self, verbose=False):
        """Initialize the parameters for the dual variables/Lagrange multipliers."""

        names = ['λ_safe', 'λ_unsafe', 'λ_cnt', 'λ_dis', 'λ_grad', 'λ_param']
        params = dict.fromkeys(names, 1.0)
        params['λ_grad'] = 0.01     # hack

        if verbose is True:
            print('Dual parameters initialization:')
            for (name, val) in params.items():
                print(f'\t * {name} = {val}')

        return params

    def _init_optimizer(self, kwargs):
        """Initialized the optimizer based on self.opt_init.
        Args:
            kwargs: Named arguments of optimization algorithm.
        Modifies:
            Creates the standard Jax triple of (init_fun, update_fun, get_params)
            opimization functions.
        """

        if self.opt_name == 'SGD':
            self.opt_init, self.opt_update, self.get_params = optimizers.sgd(**kwargs)

        elif self.opt_name == 'Adam':
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(**kwargs)

        elif self.opt_name == 'Adagrad':
            self.opt_init, self.opt_update, self.get_params = optimizers.adagrad(**kwargs)

        elif self.opt_name == 'Nesterov':
            self.opt_init, self.opt_update, self.get_params = optimizers.nesterov(**kwargs)

        else:
            opts = ['Adam', 'Adagrad', 'Nesterov', 'GD', 'SGD']
            raise NotImplementedError(f'Supported optimizers: {" | ".join(opts)}')
