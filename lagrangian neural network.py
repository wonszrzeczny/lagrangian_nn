import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import copy
print(device_lib.list_local_devices())

#  probably really slow, need to make a graph of these for future (?)
class Momentum_Generator:
    def __init__(self, axis, n, dim):
        self.array = np.zeros((n, 2, dim), dtype = np.float32)
        self.array[:, 0, axis] = 1
        self.generator_tensor = tf.convert_to_tensor(self.array)
    @tf.function
    def generator(self, x):
        return  self.generator_tensor

def body_spring(x): # easiest example dynamics
    return np.array([x[1,0,0]-x[0,0,0],x[0,0,0]-x[1,0,0]]).astype(np.float32)

def simulate(start, f, t=10, n = 100, points = 100):
    res=[]
    current_state = copy.deepcopy(start)
    dt = t/n/points
    for i in range(points):
        if i%10==0:
            print(i/10)
        for __ in range(n):
            current_state[:, 1, :] += f(current_state)[...,None]  * dt/2 # kick
            current_state[:, 0, :] += current_state[:, 1, :] * dt # drift
            current_state[:, 1, :] += f(current_state)[...,None]  * dt/2  # kick
        res.append(copy.deepcopy(current_state))
    return np.stack(res, axis=0)

def to_tensor_form(state):
    return np.stack([state[:, 0, :].flatten(), state[:, 1, :].flatten()])


class Scene: # container for physics, follows [ body index , q / v , dimensions ]

    def __init__(self, F = lambda x: x ,n = 2,d = 1):
        self.n = n # number of interacting fields
        self.d = d # space dimension
        self.F = F # dynamics function
        self.state = np.zeros((n,2,d))

        self.momentum_generator = Momentum_Generator(axis= 0, n=self.n, dim=self.d)

        self.physics_priors = [self.momentum_generator.generator]

    def ground_truth(self, state = None):

        if state != None:
            self.state=state
        return self.F(self.state)

    def evolve(self, a, n=100):  # using leapfrog integration, since it's the simplest, might upgrade to Yoshida

        for _ in range(n):
            self.state[:,1,:] += self.ground_truth()[...,None] * a/n/2   # kick
            self.state[:,0,:] += self.state[:,1,:] * a/n                 # drift
            self.state[:,1,:] += self.ground_truth()[...,None] * a/n/2   # kick

    def generate_labels(self, n):

        states = (np.random.random((n, *self.state.shape)).astype(np.float32)-0.5)*10
        labels = np.stack([self.F(state) for state in states])           # for neural net compatibility
        states = np.stack([to_tensor_form(state) for state in states])   # for neural net compatibility
        print("states", states, "labels", labels)

        return (states, labels)

    @tf.function
    def coordinate_transformation(self, x, phi):
        #   FAFF
        #   get to numpy form [batch size, particle number, position/momentum, coordinate]
        scene_form = tf.einsum("bpnd->bnpd" ,tf.reshape(x, [tf.shape(x)[0], 2, self.n, self.d]))
        #   apply transformation
        transf_grad = tf.vectorized_map(phi, scene_form)
        #   return in tensor form [batch size, position/momentum, all coordinates flat]
        return tf.reshape(tf.einsum("bnpd->bpnd", transf_grad), [tf.shape(x)[0], 2, self.n*self.d])

    @tf.function
    def prior_transformations(self, x):     # return a list of all transformations leaving the lagrangian invariant
        return [self.coordinate_transformation(x, phi) for phi in self.physics_priors]


class Neural_Net:

    def __init__(self):
        self.scene = Scene()
        self.model = None

    def create_model(self):

        # important hyper parameter, no clue what is optimal
        self.optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.loss_object = tf.keras.losses.MeanSquaredError()

        #  model architecture : input, lagrangian layer, lagrangian
        self.lagrangian_net = keras.Sequential(
            [
                layers.Input(shape = to_tensor_form(self.scene.state).shape, dtype = tf.float32),
                layers.Flatten(input_shape = to_tensor_form(self.scene.state).shape),
                layers.Dense(40, kernel_initializer='random_normal', activation='softplus', dtype = tf.float32),
                layers.Dense(40, kernel_initializer='random_normal', activation='softplus', dtype = tf.float32),
                layers.Dense(40, kernel_initializer='random_normal', activation='softplus', dtype=tf.float32),
                layers.Dense(40, kernel_initializer='random_normal', activation='softplus', dtype=tf.float32),
                layers.Dense(1, kernel_initializer='random_normal', activation='softplus', dtype = tf.float32)

            ]

        )
        self.loss_history = []

    @tf.function
    def gradient(self, x):
        with tf.GradientTape() as tape:  # observing the first order derivatives
            tape.watch(x)
            lagrangian = self.lagrangian_net(x)
        return tape.batch_jacobian(lagrangian, x, unconnected_gradients='zero')[:, 0]

    # don't change this, this one is tricky
    @tf.function
    def dynamics(self, x):  # compute guess of dynamics of the system, using current lagrangian model
        with tf.GradientTape() as tape2: # observing the first order derivatives
            tape2.watch(x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                lagrangian = self.lagrangian_net(x)
            g = tape.batch_jacobian(lagrangian, x, unconnected_gradients='zero')[:, 0]
        hessian = tape2.batch_jacobian(g, x, unconnected_gradients='zero')
        U = g[:, 0, :] - tf.einsum("dij,dj->di", hessian[:, 1, :, 0, :], x[:, 1, :])  # U[d,i]
        P = hessian[:, 1, :, 1, :]
        P = tf.map_fn(tf.linalg.inv, P)  # P[d, i, k]
        A = tf.einsum("di,dki->dk", U, P)
        return A  # return accelerations for the batch

    # for physics priors, will have to evaluate charges too, later
    @tf.function
    def lagrangian_change(self, translation_tensor, x):
        with tf.GradientTape() as tape: # observing the first order derivatives
            tape.watch(x)
            lagrangian = self.lagrangian_net(x)
        grad = tape.batch_jacobian(lagrangian, x, unconnected_gradients='zero')[:, 0]
        #   no clue how to attempt normalization here, currently comparing change of lagrangian to std
        #   might not work for smaller batch sizes, or unshufled data
        return tf.einsum("cdij,dij->cd", translation_tensor, grad)/ tf.math.reduce_std(lagrangian)

    @tf.function
    def priors_error(self, x):
        invariant_transformations = tf.stack(self.scene.prior_transformations(x), axis = 0) # stack
        print(" transformation dtype", invariant_transformations)
        return self.lagrangian_change(invariant_transformations, x)


    def generate_labels(self):  # dataset setup
        (y_true, x) = self.scene.generate_labels(100000) # add generator?
        print("nice")
        dataset = tf.data.Dataset.from_tensor_slices((y_true, x)).batch(32)
        return dataset

    def initialization_step(self, x):
        with tf.GradientTape() as tape:
            #   calculating accelerations and lagrangian changes
            constraints = self.priors_error(x)
            loss_value = self.loss_object(0, constraints)
        self.loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, self.lagrangian_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.lagrangian_net.trainable_variables))

    def initiate_priors(self):
        dataset = self.generate_labels()
        for epoch in range(40):
            for (batch, (x, true_y)) in enumerate(dataset):
                self.initialization_step(x)
            dataset.shuffle(100000, reshuffle_each_iteration=True).batch(64)
            print('Epoch {} finished'.format(epoch))
            print(self.loss_history[-1])


    # add @tf.function maybe?
    def train_step(self, x, true_y):
        with tf.GradientTape() as tape:
            #   calculating accelerations and lagrangian changes
            acc = self.dynamics(x)
            constraints = self.priors_error(x)
            acc_loss = self.loss_object(true_y, acc)
            constraint_loss = 0 # self.loss_object(0, constraints)
            total_loss = acc_loss #+ constraint_loss
        self.loss_history.append(acc_loss.numpy().mean())
        grads = tape.gradient(total_loss , self.lagrangian_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.lagrangian_net.trainable_variables))

    def train(self, epochs):
        dataset = self.generate_labels()
        #self.initiate_priors()
        for epoch in range(epochs):
            for (batch, (x, true_y)) in enumerate(dataset):
                self.train_step(x, true_y)
            dataset.shuffle(100000, reshuffle_each_iteration=True).batch(64)
            print('Epoch {} finished'.format(epoch))
            print(self.loss_history[-1][0], "acc loss", self.loss_history[-1][1], "prior loss", "\n")

    def __str__(self):
        return str(self.lagrangian_net.summary())


# physics scene
springs = np.array([ [ [0],[0] ],[ [1] , [0] ] ],dtype=np.float32)
scene = Scene(F= body_spring)
scene.state = springs


# creating the model

N = Neural_Net()
N.scene = scene
N.create_model()
N.train(30)

print("prediction")
hist = simulate(springs, body_spring)
neural_hist = simulate(springs, lambda x: N.dynamics(to_tensor_form(x)[None, ...])[0].numpy())

for item in hist:
    print(body_spring(item))
    print("acceleration function", N.dynamics(to_tensor_form(item)[None, ...])[0])


plt.plot(hist[:, 0, 0, 0])
plt.plot(neural_hist[:, 0, 0, 0])
plt.plot(neural_hist[:, 1, 0, 0])
plt.show()


