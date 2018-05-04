import gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, X, n_components = 500):
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_components = n_components
        self.X = X

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def fit(self):
        self.dimensions = [self.n_features, 250, 350, self.n_components]
        self.parameters = {}
        for n in range(len(self.dimensions) - 1):
            self.parameters["w"+str(n+1)] = np.random.randn(self.dimensions[n+1], self.dimensions[n]) * np.sqrt(1/self.dimensions[n])
            self.parameters["b"+str(n+1)] = np.ones((self.dimensions[n+1],1))

        return self

    def transform(self, S):
        n = len(self.dimensions)
        S = S.T
        A_prev = S
        for i in range(n-1):
            Z = np.dot(self.parameters["w"+str(i+1)], A_prev) + self.parameters["b" + str(i+1)]
            A = self.sigmoid(Z)
            A_prev = A

        return A

    def fit_transform(self, S):
        self.fit()
        A = self.transform(S)
        return A

class SGDRegressor:
    def __init__(self, dimension):
        self.w = np.random.randn(dimension) / np.sqrt(dimension)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        X = X.T
        self.w += self.lr*(Y - np.dot(X, self.w)).dot(X)

    def predict(self, X):
        X = X.T
        return np.dot(X, self.w)

class FeatureTransformer:
    def __init__(self, n_components = 2000):
        examples = np.random.random((20000,4))*2-2
        scaler = StandardScaler()
        scaler.fit(examples)

        Feature = NeuralNetwork(examples, n_components = n_components)
        fea_example = Feature.fit_transform(scaler.transform(examples))
        self.dimension = fea_example.shape[0]
        self.Feature = Feature
        self.scaler = scaler

    def transform(self, s):
        scaler = self.scaler.transform(np.atleast_2d(s))
        return self.Feature.transform(scaler)

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.ft = feature_transformer
        self.models = []
        for m in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimension)
            self.models.append(model)

    def predict(self, s):
        X = self.ft.transform(s)
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, a, G):
        X = self.ft.transform(s)
        self.models[a].partial_fit(X, [G])

    def random_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(env, model, ft, eps, gamma):
    observation = env.reset()
    iters = 0
    totalreward = 0
    done = False

    while not done and iters < 10000:
        action = model.random_action(observation, eps)
        old_observation = observation
        observation, reward, done, info = env.step(action)

        Q2 = model.predict(observation)
        G = reward + gamma * np.max(Q2[0])
        model.update(old_observation, action, G)

        iters += 1
        totalreward += reward

    return totalreward

def main():
    env = gym.make("CartPole-v0")
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.99
    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/(np.sqrt(n+1))
        totalreward = play_one(env, model, ft, eps, gamma)
        totalrewards[n] = totalreward

        if n % 10 == 0:
            print("Eps: ", eps)
        if (n+1) % 100 == 0:
            print("episode: ", n, "total reward:", totalreward)

    plt.plot(totalrewards)
    plt.show()


if __name__ == "__main__":
#     env = gym.make("CartPole-v0")
#     feature = FeatureTransformer()
#     obs = env.reset()
#     a = feature.transform(obs)
#     print(a)
#     print(a.shape)
    main()
