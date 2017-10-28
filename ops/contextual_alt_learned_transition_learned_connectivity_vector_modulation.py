import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import initialization


class ContextualCircuit(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            model_version='full',
            timesteps=1,
            lesions=None,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1],
            padding='SAME',
            dtype=tf.float32,
            return_weights=True):

        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in X.get_shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.strides = strides
        self.padding = padding
        self.dtype = dtype
        self.SRF, self.SSN, self.SSF = SRF, SSN, SSF

        self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, 1]
        self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.i_shape = self.q_shape
        self.o_shape = self.q_shape
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.tuning_nl = tf.nn.relu
        self.tuning_shape = [1, 1, self.k, self.k]
        self.tuning_params = ['Q', 'P', 'T']  # Learned connectivity
        self.recurrent_nl = tf.nn.relu
        self.gate_nl = tf.nn.sigmoid

        self.return_weights = return_weights
        self.normal_initializer = False
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices."""
        self.weight_dict = {  # Weights lower/activity upper
            'U': {
                'r': {
                    'weight': 'u_r',
                    'activity': 'U_r'
                    }
                },
            'T': {
                'r': {
                    'weight': 't_r',
                    'activity': 'T_r',
                    'tuning': 't_t'
                    }
                },
            'P': {
                'r': {
                    'weight': 'p_r',
                    'activity': 'P_r',
                    'tuning': 'p_t'
                    }
                },
            'Q': {
                'r': {
                    'weight': 'q_r',
                    'activity': 'Q_r',
                    'tuning': 'q_t'
                    }
                },
            'I': {
                'r': {  # Recurrent state
                    'weight': 'i_r',
                    'activity': 'I_r'
                },
                'f': {  # Recurrent state
                    'weight': 'i_f',
                    'activity': 'I_f'
                },
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'activity': 'O_r'
                },
                'f': {  # Recurrent state
                    'weight': 'o_f',
                    'activity': 'O_f'
                },
            },
            'xi': {
                'r': {  # Recurrent state
                    'weight': 'xi',
                }
            },
            'alpha': {
                'r': {  # Recurrent state
                    'weight': 'alpha',
                }
            },
            'beta': {
                'r': {  # Recurrent state
                    'weight': 'beta',
                }
            },
            'mu': {
                'r': {  # Recurrent state
                    'weight': 'nu',
                }
            },
            'zeta': {
                'r': {  # Recurrent state
                    'weight': 'zeta',
                }
            },
            'gamma': {
                'r': {  # Recurrent state
                    'weight': 'gamma',
                }
            },
            'delta': {
                'r': {  # Recurrent state
                    'weight': 'delta',
                }
            }
        }

        # tuned summation: pooling in h, w dimensions
        #############################################
        q_array = np.ones(self.q_shape) / np.prod(self.q_shape)
        setattr(
            self,
            self.weight_dict['Q']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['weight'],
                dtype=self.dtype,
                initializer=q_array.astype(np.float32),
                trainable=False)
            )

        # untuned suppression: reduction across feature axis
        ####################################################
        u_array = np.ones(self.u_shape) / np.prod(self.u_shape)
        setattr(
            self,
            self.weight_dict['U']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['U']['r']['weight'],
                dtype=self.dtype,
                initializer=u_array.astype(np.float32),
                trainable=False)
            )

        # weakly tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.ones(self.p_shape)
        p_array[
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        p_array = p_array / p_array.sum()

        setattr(
            self,
            self.weight_dict['P']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['P']['r']['weight'],
                dtype=self.dtype,
                initializer=p_array.astype(np.float32),
                trainable=False))

        # weakly tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.ones(self.t_shape)
        t_array[
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        t_array = t_array / t_array.sum()
        setattr(
            self,
            self.weight_dict['T']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['T']['r']['weight'],
                dtype=self.dtype,
                initializer=t_array.astype(np.float32),
                trainable=False))

        # Connectivity tensors -- Q/P/T
        setattr(
            self,
            self.weight_dict['Q']['r']['tuning'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['tuning'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.tuning_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['P']['r']['tuning'],
            tf.get_variable(
                name=self.weight_dict['P']['r']['tuning'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.tuning_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['T']['r']['tuning'],
            tf.get_variable(
                name=self.weight_dict['T']['r']['tuning'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.tuning_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # Input
        setattr(
            self,
            self.weight_dict['I']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['I']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['f']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # Output
        setattr(
            self,
            self.weight_dict['O']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['O']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['f']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # Vector weights
        w_array = np.ones([1, 1, 1, self.k]).astype(np.float32)
        self.xi = tf.get_variable(name='xi', initializer=w_array)
        self.alpha = tf.get_variable(name='alpha', initializer=w_array)
        self.beta = tf.get_variable(name='beta', initializer=w_array)
        self.mu = tf.get_variable(name='mu', initializer=w_array)
        self.nu = tf.get_variable(name='nu', initializer=w_array)
        self.zeta = tf.get_variable(name='zeta', initializer=w_array)
        self.gamma = tf.get_variable(name='gamma', initializer=w_array)
        self.delta = tf.get_variable(name='delta', initializer=w_array)

    def conv_2d_op(self, data, weight_key, out_key=None):
        """2D convolutions, lesion, return or assign activity as attribute."""
        if weight_key in self.lesions:
            weights = tf.constant(0.)
        else:
            weights = self[weight_key]
        activities = tf.nn.conv2d(
                data,
                weights,
                self.strides,
                padding=self.padding)
        if out_key is None:
            return activities
        else:
            setattr(
                self,
                out_key,
                activities)

    def apply_tuning(self, data, wm, nl=True):
        for k in self.tuning_params:
            if wm == k:
                data = self.conv_2d_op(
                    data=data,
                    weight_key=self.weight_dict[wm]['r']['tuning']
                    )
                if nl:
                    return self.tuning_nl(data)
                else:
                    return data
        return data

    def full(self, i0, O, I):
        """Published CM with learnable weights.
        Swap out scalar weights for GRU-style update gates:
        # Eps_eta is I forget gate
        # Eta is I input gate
        # sig_tau is O forget gate
        # tau is O input gate
        """

        # Connectivity convolutions
        U = self.conv_2d_op(
            data=self.apply_tuning(O, 'U'),
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=self.apply_tuning(O, 'T'),
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=self.apply_tuning(I, 'P'),
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=self.apply_tuning(I, 'Q'),
            weight_key=self.weight_dict['Q']['r']['weight']
        )

        # Gates
        I_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['I']['f']['weight']
        )
        I_update_recurrent = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['I']['r']['weight']
        )
        I_update = self.gate_nl(I_update_input + I_update_recurrent)
        O_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['O']['f']['weight']
        )
        O_update_recurrent = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['O']['r']['weight']
        )
        O_update = self.gate_nl(O_update_input + O_update_recurrent)

        # Circuit
        I_summand = self.recurrent_nl(
            (self.xi * self.X)
            - ((self.alpha * I + self.mu) * U)
            - ((self.beta * I + self.nu) * T))
        I = (I_update * I) + ((1 - I_update) * I_summand)
        O_summand = self.recurrent_nl(
            self.zeta * I
            + self.gamma * P
            + self.delta * Q)
        O = (O_update * O) + ((1 - O_update) * O_summand)
        i0 += 1  # Iterate loop
        return i0, O, I

    def condition(self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def gather_tensors(self, wak='weight'):
        weights = {}
        for k, v in self.weight_dict.iteritems():
            for wk, wv in v.iteritems():
                if wak in wv.keys() and hasattr(self, wv[wak]):
                    weights['%s_%s' % (k, wk)] = self[wv[wak]]

        return weights

    def build(self, reduce_memory=False):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        if reduce_memory:
            print 'Warning: Using FF version of the model.'
            for t in range(self.timesteps):
                i0, O, I = self[self.model_version](i0, O, I)
        else:
            # While loop
            elems = [
                i0,
                O,
                I
            ]

            returned = tf.while_loop(
                self.condition,
                self[self.model_version],
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)
            # Prepare output
            i0, O, I = returned  # i0, O, I
        if self.return_weights:
            weights = self.gather_tensors(wak='weight')
            tuning = self.gather_tensors(wak='tuning')
            new_tuning = {}
            for k, v in tuning.iteritems():
                key_name = v.name.split('/')[-1].split(':')[0]
                new_tuning[key_name] = v
            weights = dict(weights, **new_tuning)
            activities = self.gather_tensors(wak='activity')
            return O, weights, activities
        else:
            return O
