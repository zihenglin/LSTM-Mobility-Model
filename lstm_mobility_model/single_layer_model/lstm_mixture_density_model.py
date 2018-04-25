import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class tf_lstm_mixture_density_model_generate:

    def __init__(self,
                 name,
                 input_length,
                 n_lstm_units,
                 n_layers,
                 pred_x_dim=3,
                 obs_x_dim=2,
                 y_dim=8,
                 batch_size=1,
                 n_loc_mixtures=5,
                 n_categories=3,
                 n_finished=1,
                 n_probabilities=4,
                 dropout_prob=0.0,
                 learning_rate=0.001,
                 start_time_mean=None,
                 start_time_sd=None):

        # Model name
        self.name = name

        # NN structure
        self.input_length = input_length
        self.n_lstm_units = n_lstm_units
        self.n_layers = n_layers

        # Mixture Models
        self.n_loc_mixtures = n_loc_mixtures
        self.n_dur_mixtures = n_loc_mixtures
        self.loc_mixture_ratio = 5  # pi, mu_x, mu_y, s_x, s_y
        self.dur_mixture_ratio = 5  # mu_st, mu_dur, s_st, s_dur, rho
        self.y_dim = y_dim
        self.n_categories = n_categories
        self.n_finished = n_finished
        self.n_probabilities = n_probabilities

        # Dimensions
        self.output_dim = self.n_loc_mixtures * self.loc_mixture_ratio + \
            self.n_dur_mixtures * self.dur_mixture_ratio + \
            self.n_categories * self.n_dur_mixtures + \
            self.n_finished * self.n_dur_mixtures

        self.input_dim = self.output_dim
        self.pred_x_dim = pred_x_dim
        self.obs_x_dim = obs_x_dim
        self.obs_dim = self.obs_x_dim + self.pred_x_dim

        # Others
        self.dropout_prob = dropout_prob
        self.init_stdev = 0.075
        self.batch_size = batch_size
        self.oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
        self.learning_rate = learning_rate

        # Keep track of time
        self.time = [None] * (input_length + 1)
        self.time[0] = tf.placeholder("float", [self.batch_size, 1], name='time')

        # Build models
        self.init_model()

        # Sequence start time
        self.start_time_mean = start_time_mean
        self.start_time_sd = start_time_sd

    def init_model(self):
        with tf.variable_scope(self.name):
            # First value to get started
            self.X = tf.placeholder("float", [None, self.pred_x_dim])

            # Input value sequence
            self.input_X = tf.placeholder("float", [None, self.input_length, self.obs_x_dim])

            # True sequence for calculating loss
            self.y = tf.placeholder("float", [None, self.input_length, self.y_dim])

            # Bias for sampling
            self.location_sd_bias = tf.placeholder("float", [1], name='location_sd_bias')
            self.time_sd_bias = tf.placeholder("float", [1], name='time_sd_bias')
            self.pi_bias = tf.placeholder("float", [1], name='pi_bias')

            # Affine weights
            self.W1 = W1 = tf.Variable(
                tf.random_normal(
                    [self.n_lstm_units, self.output_dim], stddev=self.init_stdev, dtype=tf.float32))
            self.b1 = b1 = tf.Variable(
                tf.random_normal([1, self.output_dim], stddev=self.init_stdev, dtype=tf.float32))

            # Multi-layer lstm cells with dropouts
            lstm_cell = rnn_cell.BasicLSTMCell(
                self.n_lstm_units, forget_bias=0.0, state_is_tuple=True)

            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - self.dropout_prob)
            self.cell = cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell] * self.n_layers, state_is_tuple=True)

            # Init state for lstm cells
            _initial_state = cell.zero_state(self.batch_size, tf.float32)

            # Output scores
            scores = []

            # Output probabiliteis
            coefficients = []

            # Output probabiliteis
            mixture_coefficients = []

            # Predicted sequence
            gen_seq = []

            # States of lstm
            states = []

            # States of LSTM in each interations
            state = _initial_state

            # Inputs to LSTM in each interations
            obs = tf.concat(1, [self.X, self.input_X[:, 0, :]])

            cur_time = str(time.time())

            for time_step in xrange(self.input_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                (score, state) = cell(obs, state)

                # Score from affine
                score = tf.matmul(score, W1) + b1

                # Sample next point and cat with other input
                obs_pred, coef, mixture_coef = self.sample_obs(score, time_step)
                obs = tf.concat(1, [obs_pred, self.input_X[:, time_step, :]])

                # Collect scores and generated_sequences
                coefficients.append(coef)
                mixture_coefficients.append(mixture_coef)
                scores.append(score)
                gen_seq.append(obs_pred)
                states.append(state)

        scores = tf.transpose(scores, [1, 0, 2])
        coefficients = tf.transpose(coefficients, [1, 0, 2])
        mixture_coefficients = tf.transpose(mixture_coefficients, [1, 0, 2])
        gen_seq = tf.transpose(gen_seq, [1, 0, 2])

        states = tf.reshape(
            states, [self.input_length, self.batch_size, self.n_lstm_units * self.n_layers * 2])
        states = tf.transpose(states, [1, 0, 2])

        self.coefficients = coefficients
        self.mixture_coefficients = mixture_coefficients
        self.gen_seq = gen_seq
        self.states = states
        self.scores = scores

    def sample_obs(self, score, time_step):
        pi, \
            mu_locx, \
            mu_locy, \
            s_locx, \
            s_locy, \
            mu_st, \
            mu_dur, \
            s_st, \
            s_dur, \
            rho, \
            out_category, \
            finished = self.get_mixture_coef(score)

        # Sample mixture components
        mu = mu_dur + rho * s_dur / s_st * (self.time[time_step] - mu_st)
        s = tf.sqrt(1 - rho**2) * s_dur

        prob_st = self.tf_normal(
            tf.tile(self.time[time_step], [1, self.n_dur_mixtures]), mu_st, s_st)

        lob_prob = tf.log(tf.clip_by_value(pi * prob_st, 1e-30, 1e30))
        ind = tf.multinomial(lob_prob, 1)
        ind = tf.concat(1, [np.array(range(self.batch_size)).reshape(self.batch_size, 1), ind])

        # Cast to int32 in order to work with indexing requirement
        ind = tf.cast(ind, tf.int32)

        # Sample location
        sampled_x, sampled_y = self.sample_mixture_2d(ind, mu_locx, mu_locy, s_locx, s_locy,
                                                      self.n_loc_mixtures)

        sample_dur = self.sample_mixture_1d(ind, mu, s, self.n_dur_mixtures)
        cur_time = self.time[time_step]
        output = tf.concat(1, [sampled_x, sampled_y, cur_time, sample_dur])
        self.time[time_step + 1] = cur_time + sample_dur

        coef = tf.concat(1, [
            tf.reshape(tf.gather_nd(finished, ind), [self.batch_size, 1]),
            tf.gather_nd(out_category, ind)
        ])

        mixture_coef = tf.concat(1, [
            tf.reshape(tf.gather_nd(pi, ind), [self.batch_size, 1]),
            tf.reshape(tf.gather_nd(mu_locx, ind), [self.batch_size, 1]), tf.reshape(
                tf.gather_nd(mu_locy, ind), [self.batch_size, 1]), tf.reshape(
                    tf.gather_nd(s_locx, ind), [self.batch_size, 1]), tf.reshape(
                        tf.gather_nd(s_locy, ind), [self.batch_size, 1]), tf.reshape(
                            tf.gather_nd(mu_st, ind), [self.batch_size, 1]), tf.reshape(
                                tf.gather_nd(mu_dur, ind), [self.batch_size, 1]), tf.reshape(
                                    tf.gather_nd(s_st, ind), [self.batch_size, 1]), tf.reshape(
                                        tf.gather_nd(s_dur, ind), [self.batch_size, 1]), tf.reshape(
                                            tf.gather_nd(rho, ind), [self.batch_size, 1])
        ])

        return output, coef, mixture_coef

    def get_mixture_coef(self, output):

        if len(output.get_shape()) == 2:
            loc = output[:, :self.n_loc_mixtures * self.loc_mixture_ratio]

            dur = output[:, self.n_loc_mixtures * self.loc_mixture_ratio:self.n_loc_mixtures *
                         self.loc_mixture_ratio + self.n_dur_mixtures * self.dur_mixture_ratio]

            out_category = output[:, self.n_loc_mixtures * self.loc_mixture_ratio +
                                  self.n_dur_mixtures * self.dur_mixture_ratio:self.n_loc_mixtures *
                                  self.loc_mixture_ratio + self.n_dur_mixtures *
                                  self.dur_mixture_ratio + self.n_categories * self.n_loc_mixtures]

            out_finished = output[:, self.n_loc_mixtures * self.loc_mixture_ratio +
                                  self.n_dur_mixtures * self.dur_mixture_ratio + self.n_categories *
                                  self.n_loc_mixtures:]

            pi_loc, mu_locx, mu_locy, s_locx, s_locy = tf.split(1, self.loc_mixture_ratio, loc)

            mu_st, mu_dur, s_st, s_dur, rho = tf.split(1, self.dur_mixture_ratio, dur)

        elif len(output.get_shape()) == 3:
            loc = output[:, :, :self.n_loc_mixtures * self.loc_mixture_ratio]

            dur = output[:, :, self.n_loc_mixtures * self.loc_mixture_ratio:self.n_loc_mixtures *
                         self.loc_mixture_ratio + self.n_dur_mixtures * self.dur_mixture_ratio]

            out_category = output[:, :, self.n_loc_mixtures * self.loc_mixture_ratio +
                                  self.n_dur_mixtures * self.dur_mixture_ratio:self.n_loc_mixtures *
                                  self.loc_mixture_ratio + self.n_dur_mixtures *
                                  self.dur_mixture_ratio + self.n_categories * self.n_loc_mixtures]

            out_finished = output[:, :, self.n_loc_mixtures * self.loc_mixture_ratio +
                                  self.n_dur_mixtures * self.dur_mixture_ratio + self.n_categories *
                                  self.n_loc_mixtures:]

            pi_loc, mu_locx, mu_locy, s_locx, s_locy = tf.split(2, self.loc_mixture_ratio, loc)

            mu_st, mu_dur, s_st, s_dur, rho = tf.split(2, self.dur_mixture_ratio, dur)

        # Calcuate mu
        mu_locx = self.get_mu(mu_locx)
        mu_locy = self.get_mu(mu_locy)

        mu_st = self.get_non_negtive(mu_st)
        mu_dur = self.get_non_negtive(mu_dur)

        # Calculate pi
        pi_loc = self.get_pi(pi_loc, self.pi_bias)

        # Calculate s
        s_locx = self.get_non_negtive(s_locx, self.location_sd_bias)
        s_locy = self.get_non_negtive(s_locy, self.location_sd_bias)
        s_st = self.get_non_negtive(s_st, self.time_sd_bias)
        s_dur = self.get_non_negtive(s_dur, self.time_sd_bias)

        # Calculate rho
        rho = self.get_rho(rho)

        # Calculate category
        out_category = self.get_category(out_category)

        # Calculate category
        out_finished = self.get_finished(out_finished)

        return pi_loc, \
            mu_locx, \
            mu_locy, \
            s_locx, \
            s_locy, \
            mu_st, \
            mu_dur, \
            s_st, \
            s_dur, \
            rho, \
            out_category, \
            out_finished

    def generate_sequence_coefficients(self,
                                       sess,
                                       X_init,
                                       X_input_seq,
                                       start_time_list,
                                       n,
                                       location_sd_bias=0.,
                                       time_sd_bias=0.,
                                       pi_bias=0.):
        gen_seq = []
        gen_coef = []
        gen_mixture_coef = []
        gen_states = []
        for i in xrange(n):
            rand_ind = np.random.choice(X_init.shape[0], self.batch_size, replace=False)[0]

            st_sample = start_time_list[rand_ind] + \
                np.random.randn(self.batch_size).reshape(
                    self.batch_size, 1) * self.start_time_sd

            seq, coef, mixture_coef, \
                state = sess.run([self.gen_seq, self.coefficients,
                                  self.mixture_coefficients, self.states],
                                 feed_dict={self.X: [X_init[rand_ind]],
                                            self.input_X: [X_input_seq[rand_ind]],
                                            self.time[0]: st_sample,
                                            self.location_sd_bias: [location_sd_bias],
                                            self.time_sd_bias: [time_sd_bias],
                                            self.pi_bias: [pi_bias]})

            gen_seq.append(seq)
            gen_coef.append(coef)
            gen_states.append(state)
            gen_mixture_coef.append(mixture_coef)

        gen_seq = np.array(gen_seq)
        gen_seq = gen_seq.reshape(n * self.batch_size, self.input_length, self.pred_x_dim)
        gen_coef = np.array(gen_coef)
        gen_coef = gen_coef.reshape(n * self.batch_size, self.input_length, self.n_probabilities)
        gen_mixture_coef = np.array(gen_mixture_coef)
        gen_mixture_coef = gen_mixture_coef.reshape(n * self.batch_size, self.input_length, -1)
        gen_states = np.array(gen_states)
        return gen_seq, gen_coef, gen_states, gen_mixture_coef

    def get_category(self, category):
        if len(category.get_shape()) == 2:
            return tf.nn.softmax(
                tf.reshape(
                    tf.clip_by_value(category, -20, 20),
                    [self.batch_size, self.n_loc_mixtures, self.n_categories]),
                dim=-1)
        else:
            return tf.nn.softmax(
                tf.reshape(
                    tf.clip_by_value(category, -20, 20),
                    [self.batch_size, self.input_length, self.n_loc_mixtures, self.n_categories]),
                dim=-1)

    def get_finished(self, finished):
        return tf.sigmoid(tf.clip_by_value(finished, -20, 20))

    def get_pi(self, pi, bias=None):
        if bias is not None:
            pi *= 1 + bias
        return tf.clip_by_value(tf.nn.softmax(pi, dim=-1), 1e-15, 1e15)

    def get_non_negtive(self, s, bias=None):
        if bias is not None:
            s -= bias
        return tf.exp(tf.clip_by_value(s, -10, 10))

    def get_rho(self, rho):
        return tf.clip_by_value(tf.tanh(rho), -0.9999, 0.9999)

    def get_mu(self, mu):
        return tf.clip_by_value(mu, -1e5, 1e5)

    def sample_mixture_1d(self, ind, mu, s, n_mixture):
        '''
        1. Sample from multinomial
        2. Sample from Gaussian
        '''

        return tf.reshape(
            tf.clip_by_value(
                tf.random_normal([1], tf.gather_nd(mu, ind), tf.gather_nd(s, ind)), -1e20, 1e20),
            [self.batch_size, 1])

    def sample_mixture_2d(self, ind, mu_1, mu_2, s_1, s_2, n_mixture):

        return self.sample_mixture_1d(ind,
                                      mu_1,
                                      s_1,
                                      n_mixture), \
            self.sample_mixture_1d(ind,
                                   mu_2,
                                   s_2,
                                   n_mixture)

    def tf_normal(self, y, mu, sigma):

        dist = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
        return dist.pdf(y)

    def tf_2d_normal(self, x1, x2, mu1, mu2, sigma1, sigma2, n_mixtures):

        x1 = tf.tile(x1, [1, 1, n_mixtures])
        x2 = tf.tile(x2, [1, 1, n_mixtures])
        return self.tf_normal(x1, mu1, sigma1) * \
            self.tf_normal(x2, mu2, sigma2)

    def tf_2d_normal_correlation(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        mu = mu2 + rho * sigma2 / sigma1 * (x1 - mu1)
        s = tf.sqrt(1 - rho**2) * sigma2
        return self.tf_normal(x1, mu1, sigma1) * \
            self.tf_normal(x2, mu, s)


class tf_lstm_mixture_density_model_train(tf_lstm_mixture_density_model_generate):

    def __init__(self,
                 name,
                 input_length,
                 n_lstm_units,
                 n_layers,
                 pred_x_dim=3,
                 obs_x_dim=2,
                 y_dim=8,
                 batch_size=1,
                 n_loc_mixtures=5,
                 n_categories=3,
                 n_finished=1,
                 n_probabilities=4,
                 dropout_prob=0.0,
                 learning_rate=0.001,
                 start_time_mean=None,
                 start_time_sd=None):

        tf_lstm_mixture_density_model_generate.__init__(
            self,
            name=name,
            input_length=input_length,
            n_lstm_units=n_lstm_units,
            n_layers=n_layers,
            pred_x_dim=pred_x_dim,
            obs_x_dim=obs_x_dim,
            y_dim=y_dim,
            batch_size=batch_size,
            n_loc_mixtures=n_loc_mixtures,
            n_categories=n_categories,
            n_finished=n_finished,
            n_probabilities=n_probabilities,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
            start_time_mean=start_time_mean,
            start_time_sd=start_time_sd)
        self.build_loss()

    def build_loss(self):
        self.loss = self.sequence_loss(self.y, self.scores)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

    def sample_obs(self, score, time_step):
        self.time[time_step + 1] = self.y[:, time_step, 2]
        return self.y[:, time_step, :4], self.y[:, time_step], self.y[:, time_step]

    def train(self,
              X_init,
              X_input_seq,
              y,
              epochs,
              sess,
              start_time_list,
              verbose=True,
              per=100,
              location_sd_bias=0.0,
              time_sd_bias=0.0,
              pi_bias=0.0):

        total_loss = 0
        for e in xrange(epochs):
            rand_ind = np.random.choice(X_init.shape[0], self.batch_size, replace=False)[0]

            st_sample = start_time_list[rand_ind] + \
                np.random.randn(self.batch_size).reshape(
                    self.batch_size, 1) * self.start_time_sd

            _, loss = sess.run(
                [self.optimizer, self.loss],
                feed_dict={
                    self.X: [X_init[rand_ind]],
                    self.input_X: [X_input_seq[rand_ind]],
                    self.y: [y[rand_ind]],
                    self.time[0]: st_sample,
                    self.location_sd_bias: [location_sd_bias],
                    self.time_sd_bias: [time_sd_bias],
                    self.pi_bias: [pi_bias]
                })

            total_loss += loss
            if verbose and e % per == 0:
                print "Epoch: " + str(e) + " Loss: " + str(total_loss / per)
                total_loss = 0

    def sequence_loss(self, y, pred_seq):

        pi_loc, \
            mu_locx, \
            mu_locy, \
            s_locx, \
            s_locy, \
            mu_st, \
            mu_dur, \
            s_st, \
            s_dur, \
            rho, \
            out_category, \
            finished = self.get_mixture_coef(pred_seq)

        # Get mixture components
        loss = self.mdn_lossfunc(self.y[:, :, 0], self.y[:, :, 1], self.y[:, :, 2], self.y[:, :, 3],
                                 self.y[:, :, 4:7], self.y[:, :, 7:8], pi_loc, mu_locx, mu_locy,
                                 s_locx, s_locy, mu_st, mu_dur, s_st, s_dur, rho, out_category,
                                 finished)

        return tf.reduce_sum(loss)

    def mdn_lossfunc(self, loc_x, loc_y, st, dur, category, finished, pi, mu_locx, mu_locy, s_locx,
                     s_locy, mu_st, mu_dur, s_st, s_dur, rho, out_category, out_finished):

        loc_x = tf.reshape(loc_x, [self.batch_size, self.input_length, 1])
        loc_y = tf.reshape(loc_y, [self.batch_size, self.input_length, 1])
        st = tf.reshape(st, [self.batch_size, self.input_length, 1])
        dur = tf.reshape(dur, [self.batch_size, self.input_length, 1])
        category = tf.reshape(category, [self.batch_size, self.input_length, 1, self.n_categories])
        finished = tf.reshape(finished, [self.batch_size, self.input_length, self.n_finished])

        # Maskq
        mask = loc_x + loc_y + dur
        mask = tf.reshape(tf.abs(tf.sign(mask)), [self.batch_size, self.input_length])

        # Loss due to loc
        prob_loc = self.tf_2d_normal(loc_x, loc_y, mu_locx, mu_locy, s_locx, s_locy,
                                     self.n_loc_mixtures)

        # Loss due to dur
        prob_dur = self.tf_2d_normal_correlation(st, dur, mu_st, mu_dur, s_st, s_dur, rho)
        # prob_dur = self.tf_2d_normal(st, dur, mu_st, mu_dur, s_st, s_dur, self.n_loc_mixtures)

        # Loss due to category
        prob_category = tf.mul(out_category, category) + \
            tf.mul(1 - out_category, 1 - category)
        prob_category = tf.reduce_sum(prob_category, -1)

        # Loss due to category
        prob_finished = (tf.mul(out_finished, finished) + tf.mul(1 - out_finished, 1 - finished))

        # Total loss
        prob = prob_loc * prob_dur * prob_category * prob_finished * pi
        prob = tf.reduce_sum(prob, -1)
        prob = -tf.log(tf.clip_by_value(prob, 1e-30, 1e30))
        prob *= mask
        loss = tf.reduce_sum(prob)
        return loss / self.batch_size
