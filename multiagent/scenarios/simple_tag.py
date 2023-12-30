import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 3
        num_adversaries = 6
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.04 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0  # 追逐的智能体的加速度要更小
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3  # 追逐的智能体的最大速度也更小，更加符合
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.10
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)

        for i, agent in enumerate(world.agents):
            if not agent.adversary:
                agent.state.p_pos = np.array([-0.8, -0.6 + (i - 6) * 0.6])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     shape = True
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 10
    #     return rew

    # def adversary_reward(self, agent, world):
    #     """
    #     距离惩罚、碰撞惩罚、捕获奖励、出界惩罚
    #     """
    #     dis_reward = 0  # 距离惩罚
    #     # col_reward = 0  # 碰撞惩罚
    #     # cap_reward = 0  # 捕获奖励
    #     other_all_reward = 0
    #     all_reward = 0
    #     kl_reward = 0  # 分布惩罚
    #     bound_reward = 0
    #
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #
    #     differ_distribution = np.zeros((len(agents),))
    #     need_transition = np.zeros((len(agents),))
    #     land = np.zeros((len(agents),))
    #
    #     for a in adversaries:
    #         dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in agents]
    #         min_index = dist.index(min(dist))
    #         land[min_index] += 1
    #
    #
    #     for adv in adversaries:  # 距离惩罚
    #         dis_reward -= min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
    #
    #     # if agent.collide:  # 和障碍物的碰撞惩罚
    #     #     for lan in world.landmarks:
    #     #         if self.is_collision(lan, agent):
    #     #             col_reward -= 1
    #     #
    #     # if agent.collide:  # 和目标的碰撞奖励
    #     #     for ag in agents:
    #     #         for adv in adversaries:
    #     #             if self.is_collision(ag, adv):
    #     #                 cap_reward += 10
    #
    #     total_sum = sum(land)
    #     normalized_land = np.array([value / total_sum for value in land])
    #     differ_distribution = world.target_distribute - normalized_land
    #     need_transition = np.round(differ_distribution * total_sum)
    #
    #     kl_reward = -np.sum(np.abs(need_transition)) * 5  # 分布惩罚
    #
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 50
    #         return min(np.exp(2 * x - 2), 30)
    #
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         bound_reward -= bound(x)
    #
    #     # print("dis_reward:{}, col_reward:{}, cap_reward:{}, kl_reward:{}" .
    #     #       format(dis_reward, col_reward, cap_reward, kl_reward))
    #     dis_reward = dis_reward * 2
    #     print("target_distribute:{}".format(world.target_distribute))
    #     print("other_reward:{}, dis_reward:{}, kl_reward:{}, bound_reward:{}".format(other_all_reward,
    #                                                                         dis_reward, kl_reward, bound_reward))
    #     all_reward = other_all_reward + kl_reward + bound_reward
    #
    #     return all_reward

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         if not other.adversary:
    #             other_vel.append(other.state.p_vel)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        """
        与每一个敌方的相对位置、与每一个己方的相对位置、每一个敌方的速度信息、各个agent与各个敌方目标的距离、与landmark的相对位置
        目标分布、当前分布、需要转移的数目
        """
        entity_pos = []  # 与landmark的相对位置
        other_pos = []  # 与其他agent的相对位置
        other_vel = []  # 敌方agent的速度信息
        adv_good_dis = []  # 我方每一个agent与敌方的距离

        agents = self.good_agents(world)  # good agents
        adversaries = self.adversaries(world)  # adversaries

        differ_distribution = np.zeros((len(agents), ))
        need_transition = np.zeros((len(agents), ))
        land = np.zeros((len(agents), ))

        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        for a in adversaries:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in agents]
            min_index = dist.index(min(dist))
            land[min_index] += 1
            adv_good_dis.append(dist)

        self_pos = agent.state.p_pos  # 自己的位置与速度
        self_vel = agent.state.p_vel

        # 下面计算应该转移的数目
        total_sum = sum(land)
        normalized_land = np.array([value / total_sum for value in land])
        differ_distribution = world.target_distribute - normalized_land
        need_transition = np.round(differ_distribution * total_sum)

        adv_good_dis_array = np.concatenate(adv_good_dis).reshape(-1)
        # 如果是good agent

        if not agent.adversary:
            return np.concatenate(
                [agent.state.p_vel.ravel()] +  # 我方智能体的速度
                [agent.state.p_pos.ravel()] +  # 我方智能体的位置
                [pos.ravel() for pos in other_pos] +  # 与其他我方智能体的相对位置
                [vel.ravel() for vel in other_vel]  # 敌方智能体的速度信息
            )
        else:
            return np.concatenate(
                [agent.state.p_vel.ravel()] +  # 敌方智能体的速度
                [agent.state.p_pos.ravel()] +  # 敌方智能体的位置
                [pos.ravel() for pos in other_pos] +  # 与其他敌方智能体的相对位置
                [vel.ravel() for vel in other_vel] +  # 我方智能体的速度信息
                [adv_good_dis_array] +  # 我方智能体与敌方智能体之间的距离信息
                [need_transition.ravel()] +
                [normalized_land.ravel()] +
                [world.target_distribute.ravel()]
            )

    def adversary_reward(self, agent, world):
        """
        奖励构成：距离奖励、分布奖励、碰撞惩罚、出界惩罚
        """
        dis_reward = 0
        kl_reward = 0
        collide_reward = 0

        agents = self.good_agents(world)  # good agents
        adversaries = self.adversaries(world)  # adversaries
        differ_distribution = np.zeros((len(agents),))
        need_transition = np.zeros((len(agents),))
        land = np.zeros((len(agents),))

        for adv in adversaries:  # 距离惩罚
            dis_reward -= min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])

        for a in adversaries:
            if self.is_collision(a, agent):
                collide_reward -= 1
                
        if abs(agent.state.p_pos[0]) > 1.5 or abs(agent.state.p_pos[1]) > 1.5:
            bound_reward = -100
        else:
            bound_reward = 0

        for a in adversaries:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in agents]
            min_index = dist.index(min(dist))
            land[min_index] += 1
        total_sum = sum(land)
        normalized_land = np.array([value / total_sum for value in land])
        differ_distribution = world.target_distribute - normalized_land
        need_transition = np.round(differ_distribution * total_sum)
        kl_reward = -np.sum(np.abs(need_transition)) * 4  # 分布惩罚

        # print("kl_reward:{}, dis_reward:{}, collide_reward:{}".format(kl_reward, dis_reward, collide_reward))
        reward = kl_reward + dis_reward + collide_reward + bound_reward

        return reward
    def get_done(self, agent, world):
        """
        当有两个good agent跑出界时，开始下一回合
        """
        if abs(agent.state.p_pos[0]) > 1.5 or abs(agent.state.p_pos[1]) > 1.5:
            return True
        else:
            return False

        # adversaries = self.adversaries(world)  # adversaries
        # bound_done = 0
        # for a in adversaries:
        #     if abs(a.state.p_pos[0]) > 1.5 or abs(a.state.p_pos[1]) > 1.5:
        #         bound_done += 1
        # if bound_done > len(adversaries) / 2:
        #     return True
        # else:
        #     return False



