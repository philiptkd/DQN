from grid_env import GridEnv
from experience_replay import ExperienceReplay
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

episodes = 1000
runs = 10
gamma = .95
update_every = 100

class DQN_agent():
    def __init__(self):
        self.eps = 0.1
        self.env = GridEnv(3)
        self.replay = ExperienceReplay(10000)   # passing size of buffer
        self.batch_size = 20

        # define graph
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.env.state_size))
        self.target_values = tf.placeholder(tf.float32, shape=(None, ))
        self.actions = tf.placeholder(tf.int32, shape=(None, ))
        self.Q_out_op, self.Q_update_op = self.build_graph()    # build main network
        self.target_Q_out_op, _ = self.build_graph('target')    # build identical target network
        
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()


    def build_graph(self, scope='main'):
        with tf.variable_scope(scope):
            h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, name="h")
            outputs = tf.layers.dense(h, self.env.num_actions, activation=tf.nn.softmax, name="outputs")
            
            # nonzero error only for selected actions
            action_mask = tf.one_hot(self.actions, self.env.num_actions, on_value=True, off_value=False)
            targets = tf.tile(tf.expand_dims(self.target_values,1), [1,self.env.num_actions])
            target_outputs = tf.where(action_mask, targets, outputs)    # takes target value where mask is true. takes outputs value otherwise
            loss = tf.reduce_sum(tf.square(target_outputs - outputs))
            update = tf.train.AdamOptimizer().minimize(loss)
        return outputs, update


    def train(self):
        steps_per_ep = np.zeros(episodes)
        for episode in range(episodes):
            print(episode)
            self.env.reset() 
            state = self.env.state
            done = False
            num_steps = 0
            while not done:
                num_steps += 1
                action = self.get_eps_action(state, self.eps)
                next_state, reward, done, _ = self.env.step(action)
                self.replay.add((state, action, reward, next_state, done))    # store in experience replay
                minibatch = self.replay.sample(self.batch_size)    # sample from experience replay
                self.net_update(minibatch)  # qlearning
                if num_steps%update_every == 0:
                    self.target_net_update()    # slowly update target network
                state = next_state
            steps_per_ep[episode] = num_steps
        return steps_per_ep


    # from https://tomaxent.com/2017/07/09/Using-Tensorflow-and-Deep-Q-Network-Double-DQN-to-Play-Breakout/
    def target_net_update(self):
        # get sorted lists of parameters in each of the networks
        main_params = [t for t in tf.trainable_variables() if t.name.startswith("main")]
        main_params = sorted(main_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables() if t.name.startswith("target")]
        target_params = sorted(target_params, key=lambda v: v.name)

        update_ops = []
        for main_v, target_v in zip(main_params, target_params):
            op = target_v.assign(main_v)
            update_ops.append(op)

        self.sess.run(update_ops)


    # minibatch qlearning
    def net_update(self, minibatch):
        states, actions, rewards, next_states, dones = minibatch
        not_dones = np.logical_not(dones)

        # create a shape (batch_size, ) array of target values
        target_values = rewards.astype(float) # np.array of shape (batch_size, )
        next_inputs = next_states[not_dones]    # np.array of shape (#done, state_size)
        next_Qs = self.sess.run(self.Q_out_op, {self.inputs: next_inputs})  # np.array of shape (#done, num_actions)
        max_Qs = np.max(next_Qs, axis=1)    # np.array of shape (#done,)
        target_values[not_dones] += gamma*max_Qs

        # compute gradients and update parameters
        self.sess.run(self.Q_update_op, {self.inputs: states, self.target_values: target_values, self.actions: actions})


    # returns eps-greedy action with respect to Q
    def get_eps_action(self, state, eps):
        if self.env.np_random.uniform() < eps:
            action = self.env.sample()
        else:
            Q = self.sess.run(self.Q_out_op, {self.inputs: np.array([state])})
            max_actions = np.where(np.ravel(Q) == Q.max())[0]
            action = self.env.np_random.choice(max_actions) # to select argmax randomly
        return action


avg_steps = np.zeros(episodes)
agent = DQN_agent()
for run in range(runs):
    print("run: ", run)
    agent.sess.run(agent.init_op)
    avg_steps += (agent.train() - avg_steps)/(run+1)
plt.plot(avg_steps)
plt.xlabel("Episode")
plt.ylabel("Average Number of Steps")
plt.show()
