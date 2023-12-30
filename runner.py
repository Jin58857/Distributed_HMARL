from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def update_target_distribute(self):
        """
        更新目标分布，从而可以实现人为指定分布
        """
        random_integers = np.random.randint(0, 10, size=3)
        total = np.sum(random_integers)

        # 处理除零情况
        if total == 0:
            ratios = np.zeros_like(random_integers)
        else:
            ratios = random_integers / total

        ratio_num = ratios * 6.0  # 将 6 修改为 6.0，确保使用浮点数
        rounded_array = np.round(ratio_num).astype(np.int64)

        # Check if the sum of rounded_array is 6 and adjust if necessary
        while np.sum(rounded_array, dtype=np.int64) != 6:
            diff = np.sum(rounded_array, dtype=np.int64) - 6
            if diff > 0:
                # Find indices of the maximum elements and decrease them
                indices = np.flatnonzero(rounded_array == rounded_array.max())
                decrease_index = np.random.choice(indices)
                rounded_array[decrease_index] -= 1
            elif diff < 0:
                # Find indices of the minimum elements and increase them
                indices = np.flatnonzero(rounded_array == rounded_array.min())
                increase_index = np.random.choice(indices)
                rounded_array[increase_index] += 1

        # Now rounded_array should sum up to 6
        target_distribute = rounded_array / 6.0  # 将 6 修改为 6.0，确保使用浮点数
        return target_distribute

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_players):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        count = 0
        for episode in tqdm(range(self.args.max_episode)):
            # reset the environment
            s = self.env.reset()
            self.env.world.target_distribute = self.update_target_distribute()  # 更新分布
            for time_step in range(self.args.max_episode_len):
                count += 1
                # self.env.render()
                actions = []
                u = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents[:6]):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                    for agent_id, agent in enumerate(self.agents[-3:]):
                        actions.append([0, 0.05, 0, 0, 0])
                s_next, r, done, info = self.env.step(actions)
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                          s_next[:self.args.n_agents])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents[:self.args.n_agents]:
                        other_agents = self.agents[:self.args.n_agents].copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)

                count_true = np.count_nonzero(done[:6])  # 检测探索时我方出界的个数
                if count_true > 5:
                    break

                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.epsilon - 0.0000005)

                if count > 0 and count % self.args.evaluate_rate == 0:
                    returns.append(self.evaluate())
                    plt.figure()
                    plt.plot(range(len(returns)), returns)
                    plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                    plt.ylabel('average returns')
                    plt.savefig(self.save_path + '/plt.png', format='png')
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            self.env.world.target_distribute = self.update_target_distribute()  # 更新分布
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents[:6]):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        actions.append(action)
                    for agent_id, agent in enumerate(self.agents[-3:]):
                        # action = agent.select_action(s[agent.agent_id], 0, 0)
                        # actions.append(action)
                        actions.append([0, 0.05, 0, 0, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
