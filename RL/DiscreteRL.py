import numpy as np
import random


class Agent:
    def __init__(self, gamma, state, action):
        self.gamma = gamma
        self.state = state
        self.action = action
        self.Q = np.zeros((state, action))
        self.log = []


class SarsaAgent(Agent):
    def reset(self):
        self.Q[:] = 0

    def e_predictAction(self, state, epsilon):
        p = np.zeros(self.action)
        p[:] = epsilon / self.action
        p[np.argmax(self.Q[state])] += 1 - epsilon
        return np.random.choice(range(self.action), p=p)

    def predictAction(self, state, epsilon):
        return self.e_predictAction(state, epsilon)

    def train(self, env, alpha=0.8, epoch=100, epsilon=0.01):
        self.alpha = alpha
        for i in range(epoch):
            score = 0
            state = env.reset()[0]
            action = self.e_predictAction(state, epsilon)
            done = False
            while not done:
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.e_predictAction(next_state, epsilon)
                self.learn(state, action, reward, next_state, next_action)
                score += reward
                state = next_state
                action = next_action
            self.log.append(score)

    def learn(self, state, action, reward, next_state, next_action):
        self.Q[state, action] += self.alpha * (
            reward
            + self.gamma * self.Q[next_state, next_action]
            - self.Q[state, action]
        )

    def run(self, env, epsilon=0.01):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = self.predictAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
        return score


class QLearningAgent(Agent):
    def reset(self):
        self.Q[:] = 0

    def e_predictAction(self, state, epsilon):
        p = np.zeros(self.action)
        p[:] = epsilon / self.action
        p[np.argmax(self.Q[state])] += 1 - epsilon
        return np.random.choice(range(self.action), p=p)

    def predictAction(self, state, epsilon):
        return np.argmax(self.Q[state])

    def train(self, env, alpha=0.8, epoch=100, epsilon=0.01):
        self.alpha = alpha
        for i in range(epoch):
            state = env.reset()[0]
            done = False
            score = 0
            while not done:
                action = self.e_predictAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)
                score += reward
                self.learn(state, action, reward, next_state)
                state = next_state
            self.log.append(score)

    def learn(self, state, action, reward, next_state):
        self.Q[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        )

    def run(self, env, epsilon=0.01):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = self.predictAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
        return score


class DynaQLearningAgent(Agent):
    def __init__(self, gamma, state, action):
        super().__init__(gamma, state, action)
        self.model = {}
        self.states = set()
        self.actions = set()

    def reset(self):
        self.Q[:] = 0
        self.model = {}

    def e_predictAction(self, state, epsilon):
        p = np.zeros(self.action)
        p[:] = epsilon / self.action
        p[np.argmax(self.Q[state])] += 1 - epsilon
        return np.random.choice(range(self.action), p=p)

    def predictAction(self, state, epsilon):
        return np.argmax(self.Q[state])

    def train(self, env, alpha=0.8, epoch=100, epsilon=0.01):
        self.alpha = alpha
        for i in range(epoch):
            state = env.reset()[0]
            done = False
            score = 0
            while not done:
                action = self.e_predictAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)
                self.model[(state, action)] = (reward, next_state)
                self.actions.add(action)
                self.states.add(state)
                score += reward
                self.learn(state, action, reward, next_state)
                self.plan(10)
                state = next_state
            self.log.append(score)

    def learn(self, state, action, reward, next_state):
        self.Q[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        )

    def plan(self, N: int):
        for i in range(N):
            S = random.choice(list(self.states))
            A = random.choice(list(self.actions))
            R, N_S = self.model.get((S, A), (0, 0))
            self.Q[S, A] += self.alpha * (
                R + self.gamma * np.max(self.Q[N_S]) - self.Q[S, A]
            )

    def run(self, env, epsilon=0.01):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = self.predictAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
        return score
