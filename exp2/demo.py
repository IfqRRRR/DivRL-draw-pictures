import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class LinUCB:
    def __init__(self, arm_num, state_dim):
        self.algo_name = 'linucb'
        self.alpha = 0.025
        self.d = state_dim  # dimension of user features
        self.arm_num = arm_num
        self.Aa = [np.identity(self.d)] * arm_num # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = [np.identity(self.d)] * arm_num  # AaI : store the inverse of all Aa matrix
        self.ba = [np.zeros(self.d)] * arm_num  # ba : collection of vectors to compute disjoin part, d*1
        self.theta = [np.zeros(self.d)] * arm_num

    def update(self, r):
        temp_x = self.x[:, None]
        self.Aa[self.chosen_arm] += np.dot(temp_x, np.transpose(temp_x))
        self.ba[self.chosen_arm] += r * self.x
        self.AaI[self.chosen_arm] = np.linalg.inv(self.Aa[self.chosen_arm])
        self.theta[self.chosen_arm] = np.dot(self.AaI[self.chosen_arm], self.ba[self.chosen_arm])

    def recommend(self, state):
        p = []
        for arm in range(self.arm_num):
            pa = np.dot(state[arm], self.theta[arm]) + self.alpha * np.sqrt(np.dot(np.dot(self.AaI[arm], state[arm]), state[arm]))
            p.append(pa)
        self.chosen_arm = np.argmax(p)
        # self.chosen_arm = np.random.randint(self.arm_num)
        self.x = state[self.chosen_arm]
        return self.chosen_arm

class UniformSample:
    def __init__(self, arm_num):
        self.algo_name = 'uniform'
        self.arm_num = arm_num

    def update(self, r):
        pass

    def recommend(self, state):
        return np.random.randint(self.arm_num)

class IterationSample:
    def __init__(self, arm_num):
        self.algo_name = 'iteration'
        self.arm_num = arm_num
        self.counting = 1

    def update(self, r):
        if r != 1:
            self.counting += 1

    def recommend(self, state):
        if self.counting >= self.arm_num:
            return self.arm_num - 1
        return self.counting

class UCB:
    def __init__(self, arm_num):
        self.algo_name = 'ucb'
        self.count = [0] * arm_num
        self.all_count = 0
        self.reward = [0] * arm_num
        self.arm_num = arm_num
        self.alpha = 0.5

    def update(self, r):
        self.reward[self.chosen_arm] += r
        self.count[self.chosen_arm] += 1
        self.all_count += 1

    def recommend(self, state):
        max_value = -100
        for arm in range(self.arm_num):
            value = self.reward[arm] / (self.count[arm] + 0.1) + self.alpha * np.sqrt(np.log(self.all_count + 1) / (self.count[arm] + 0.1))
            if value > max_value:
                max_value = value
                self.chosen_arm = arm
        return self.chosen_arm
        

def get_distance(dis):
    distance = []
    for single_state in dis:
        distance.append(np.linalg.norm(single_state, 2))
    return np.mean(distance)

class Env:
    def __init__(self, arm_num, state_dim, max_distance):
        self.state = np.random.random([arm_num, state_dim])
        self.state[0] = np.array([0.5, 0.5])
        self.initial_state = deepcopy(self.state)
        self.stride = 1e-2
        self.max_distance = max_distance / 10

    def reset(self):
        self.state = deepcopy(self.initial_state)

    def step(self, arm, algo_name):
        center = np.mean(self.state, 0)

        if not self.checkvalid(arm):
            valid = False
        elif algo_name == 'iteration':
            pre_state = self.state[:arm + 1]
            pre_center = np.mean(pre_state, 0)
            if np.linalg.norm(self.state[arm] - pre_center, 2) > self.max_distance:
                valid = False
            else:
                valid = True
        elif np.linalg.norm(self.state[arm] - center, 2) > self.max_distance:
            valid = False
        else:
            valid = True

        if valid:
            reward = 1
            direction = self.state[arm] - center
            stride = self.stride
        else:
            reward = 0
            direction = np.random.normal(loc=0.0, scale=0.1, size=self.state.shape[1])
            stride = self.stride / 2
        self.state[arm] += stride * direction / np.linalg.norm(direction, 2)
        center = np.mean(self.state, 0)
        dis = self.state - center
        distance = get_distance(dis)
            
        return distance, reward

    def checkvalid(self, arm):
        for state_dim in range(self.state.shape[1]):
            if self.state[arm][state_dim] < 0 or self.state[arm][state_dim] > 1:
                return False
        return True

seed = 1
np.random.seed(seed)
COLORS = ['blue', 'green', 'yellow']
arm_num = 8
state_dim = 2
max_distance = 10
epochs = 400
linucb = LinUCB(arm_num, state_dim)
ucb = UCB(arm_num)
uniformsample = UniformSample(arm_num)
iterationsample = IterationSample(arm_num)
algos = [ucb, uniformsample, iterationsample]
algo = linucb
# algo = ucb
# algo = uniformsample
# algo = iterationsample
env = Env(arm_num, state_dim, max_distance)
for algo_index, algo in enumerate(algos):
    env.reset()
    distances = []
    for _ in range(epochs):
        chosen_arm = algo.recommend(env.state)
        distance, reward = env.step(chosen_arm, algo.algo_name)
        distances.append(distance)
        algo.update(reward)
    with open('output_{}_{}.txt'.format(arm_num, max_distance), 'a') as file:
        file.write(str(distances))
        file.write('\n')
    xtick = np.arange(epochs)
    plt.plot(xtick, distances, c=COLORS[algo_index])
plt.savefig('final.png')
    
