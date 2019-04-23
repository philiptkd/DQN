from grid_env import GridEnv
from experience_replay import ExperienceReplay
import numpy as np
import tensorflow as tf

episodes = 1000
runs = 1
gamma = .95
alpha = 0.001   # rate at which the target net tracks the main net

class DQN_agent():
    def __init__(self):
        self.env = GridEnv(6)
        self.eps = 0.1
        self.state_size = 2 # input is tuple of coordinates (x,y)
        self.replay = ExperienceReplay(10000)   # passing size of buffer
        self.batch_size = 10

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.state_size))
        self.target_values = tf.placeholder(tf.float32, shape=(None, self.env.num_actions))
        self.actions = tf.placeholder(tf.float32, shape=(None, ))
        self.Q_out_op, self.Q_update_op = self.build_graph()    # build main network
        self.target_Q_out_op, _ = self.build_graph('target')    # build identical target network
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        

    def build_graph(self, scope='main'):
        with tf.variable_scope(scope):
            h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, name="h")
            outputs = tf.layers.dense(h, self.env.num_actions, activation=tf.nn.softmax, name="outputs")
            
            # nonzero error only for selected actions
            targets = outputs   # tensor of shape (batch_size, num_actions)
            batch_idxs = tf.expand_dims(tf.range(self.batch_size),1)    # tensor of shape (batch_size, 1)
            action_idxs = tf.expand_dims(self.actions, 1)   # tensor of shape (batch_size, 1)
            indices = tf.concat([batch_idxs, action_idxs],1)    # tensor of shape (batch_size, 2)
            targets = tf.scatter_nd_update(targets, indices, self.target_values) # targets[indices] = target_values

            loss = tf.reduce_sum(tf.square(targets - outputs))
            update = tf.train.AdamOptimizer().minimize(loss)
        return outputs, update

    def train(self):
        for episode in episodes:
            self.env.reset() # initializes state randomly
            state = env.state
            done = False
            while not done:
                action, Q_out = self.get_eps_action(state, self.eps)
                next_state, reward, done, _ = env.step(action)
                self.replay.add((state, action, reward, next_state, done))    # store in experience replay
                minibatch = self.replay.sample(self.batch_size)    # sample from experience replay
                self.net_update(minibatch)  # qlearning
                self.target_net_update()    # slowly update target network
                state = next_state

    # minibatch qlearning
    def net_update(self, minibatch):
        states, actions, rewards, next_states, dones = minibatch
        not_dones = np.logical_not(dones)

        # create a size (batch_size, ) array of target values
        target_values = rewards # np.array of size (batch_size, )
        next_inputs = next_states[not_dones]    # np.array of size (#done, state_size)
        next_Qs = self.sess.run(self.Q_out_op, {self.inputs: next_inputs})  # np.array of size (#done, num_actions)
        max_Qs = np.max(next_Qs, axis=1)    # np.array of size (#done,)
        target_values[not_dones] += gamma*max_Qs

        # compute gradients and update parameters
        self.sess.run(self.Q_update_op, {self.inputs: states, self.target_values: target_values, self.actions: actions})

    # returns eps-greedy action with respect to Q
    def get_eps_action(self, state, eps):
        Q = self.sess.run(self.Q_out_op, {self.inputs: state})
        if self.env.np_random.uniform() < eps:
            action = self.env.sample()
        else:
            max_actions = np.where(np.ravel(Q) == Q.max())[0]
            action = self.env.np_random.choice(max_actions) # to select argmax randomly
        return action, Q

