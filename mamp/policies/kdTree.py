import numpy as np
import time
from math import sqrt, sin, cos, atan2, asin, pi, floor, acos, asin
from itertools import combinations, product
from mamp.agents.agent import Agent, Obstacle
from mamp.envs import Config
from mamp.util import l2norm, leftOf, det, l2normsq, sqr


class AgentTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.left = 0
        self.right = 0


class ObstacleTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.left = 0
        self.right = 0


class KDTree(object):
    def __init__(self, agents, obstacles):
        self.agent_tree_node = AgentTreeNode()
        self.obstacle_tree_node = ObstacleTreeNode()
        self.agentTree = []
        for i in range(2 * len(agents) - 1):
            self.agentTree.append(AgentTreeNode())
        self.agents = agents
        self.obstacles = obstacles
        self.agentIDs = []
        for obj in agents:
            self.agentIDs.append(obj.id)
        self.obstacleTree = []
        self.obstacleIDs = []
        for obj in obstacles:
            self.obstacleTree.append(ObstacleTreeNode())
            self.obstacleIDs.append(obj.id)
        self.max_leaf_size = 10
        self.epsilon = 1e-5

    def buildAgentTreeRecursive(self, begin, end, node):
        # print(node)
        self.agentTree[node].begin = begin
        self.agentTree[node].end = end

        self.agentTree[node].minX = self.agentTree[node].maxX = self.agents[self.agentIDs[begin]].pos_global_frame[0]
        self.agentTree[node].minY = self.agentTree[node].maxY = self.agents[self.agentIDs[begin]].pos_global_frame[1]
        for i in range(begin, end):
            if self.agents[self.agentIDs[i]].pos_global_frame[0] > self.agentTree[node].maxX:
                self.agentTree[node].maxX = self.agents[self.agentIDs[i]].pos_global_frame[0]
            elif self.agents[self.agentIDs[i]].pos_global_frame[0] < self.agentTree[node].minX:
                self.agentTree[node].minX = self.agents[self.agentIDs[i]].pos_global_frame[0]
            if self.agents[self.agentIDs[i]].pos_global_frame[1] > self.agentTree[node].maxY:
                self.agentTree[node].maxY = self.agents[self.agentIDs[i]].pos_global_frame[1]
            elif self.agents[self.agentIDs[i]].pos_global_frame[1] < self.agentTree[node].minY:
                self.agentTree[node].minY = self.agents[self.agentIDs[i]].pos_global_frame[1]

        if end - begin > self.max_leaf_size:  # no leaf node      max_leaf_size=10
            vertical = (self.agentTree[node].maxX - self.agentTree[node].minX) > (
                    self.agentTree[node].maxY - self.agentTree[node].minY)  # vertical split
            splitValue = 0.5 * (self.agentTree[node].maxX + self.agentTree[node].minX) if vertical else 0.5 * (
                    self.agentTree[node].maxY + self.agentTree[node].minY)

            l = begin  # left
            r = end - 1  # right
            while True:
                while (l <= r) and ((self.agents[self.agentIDs[l]].pos_global_frame[0]
                if vertical else self.agents[self.agentIDs[l]].pos_global_frame[1]) < splitValue):
                    l += 1
                while (r >= l) and ((self.agents[self.agentIDs[r]].pos_global_frame[0]
                if vertical else self.agents[self.agentIDs[r]].pos_global_frame[1]) >= splitValue):
                    r -= 1
                if l > r:
                    break
                else:
                    self.agentIDs[l], self.agentIDs[r] = self.agentIDs[r], self.agentIDs[l]
                    l += 1
                    r -= 1

            leftsize = l - begin

            if leftsize == 0:
                leftsize += 1
                l += 1
                r += 1

            self.agentTree[node].left = node + 1
            self.agentTree[node].right = node + 1 + (2 * leftsize - 1)

            self.buildAgentTreeRecursive(begin, l, self.agentTree[node].left)
            self.buildAgentTreeRecursive(l, end, self.agentTree[node].right)

    def buildAgentTree(self):
        if self.agentIDs:
            self.buildAgentTreeRecursive(0, len(self.agentIDs), 0)

    def queryAgentTreeRecursive(self, agent, rangeSq, node):
        if self.agentTree[node].end - self.agentTree[node].begin <= self.max_leaf_size:
            for i in range(self.agentTree[node].begin, self.agentTree[node].end):
                agent.insertAgentNeighbor(self.agents[self.agentIDs[i]], rangeSq)
        else:
            distSqLeft = 0.0
            distSqRight = 0.0
            if agent.pos_global_frame[0] < self.agentTree[self.agentTree[node].left].minX:
                distSqLeft += sqr(self.agentTree[self.agentTree[node].left].minX - agent.pos_global_frame[0])
            elif agent.pos_global_frame[0] > self.agentTree[self.agentTree[node].left].maxX:
                distSqLeft += sqr(agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].left].maxX)
            if agent.pos_global_frame[1] < self.agentTree[self.agentTree[node].left].minY:
                distSqLeft += sqr(self.agentTree[self.agentTree[node].left].minY - agent.pos_global_frame[1])
            elif agent.pos_global_frame[1] > self.agentTree[self.agentTree[node].left].maxY:
                distSqLeft += sqr(agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].left].maxY)
            if agent.pos_global_frame[0] < self.agentTree[self.agentTree[node].right].minX:
                distSqRight += sqr(self.agentTree[self.agentTree[node].right].minX - agent.pos_global_frame[0])
            elif agent.pos_global_frame[0] > self.agentTree[self.agentTree[node].right].maxX:
                distSqRight += sqr(agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].right].maxX)
            if agent.pos_global_frame[1] < self.agentTree[self.agentTree[node].right].minY:
                distSqRight += sqr(self.agentTree[self.agentTree[node].right].minY - agent.pos_global_frame[1])
            elif agent.pos_global_frame[1] > self.agentTree[self.agentTree[node].right].maxY:
                distSqRight += sqr(agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].right].maxY)

            if distSqLeft < distSqRight:
                if distSqLeft < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)
                if distSqRight < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)
            else:
                if distSqRight < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)
                if distSqLeft < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)

    def computeAgentNeighbors(self, agent, rangeSq):
        self.queryAgentTreeRecursive(agent, rangeSq, 0)

    def deleteObstacleTree(self, node):
        if node.obstacleID == -1:
            del node
        else:
            self.deleteObstacleTree(node.left)
            self.deleteObstacleTree(node.right)
            del node

    def buildObstacleTreeRecursive(self, begin, end, node):
        self.obstacleTree[node].begin = begin
        self.obstacleTree[node].end = end

        self.obstacleTree[node].minX = self.obstacleTree[node].maxX = self.obstacles[self.obstacleIDs[begin]].pos[0]
        self.obstacleTree[node].minY = self.obstacleTree[node].maxY = self.obstacles[self.obstacleIDs[begin]].pos[1]
        for i in range(begin, end):
            if self.obstacles[self.obstacleIDs[i]].pos[0] > self.obstacleTree[node].maxX:
                self.obstacleTree[node].maxX = self.obstacles[self.obstacleIDs[i]].pos[0]
            elif self.obstacles[self.obstacleIDs[i]].pos[0] < self.obstacleTree[node].minX:
                self.obstacleTree[node].minX = self.obstacles[self.obstacleIDs[i]].pos[0]
            if self.obstacles[self.obstacleIDs[i]].pos[1] > self.obstacleTree[node].maxY:
                self.obstacleTree[node].maxY = self.obstacles[self.obstacleIDs[i]].pos[1]
            elif self.obstacles[self.obstacleIDs[i]].pos[1] < self.obstacleTree[node].minY:
                self.obstacleTree[node].minY = self.obstacles[self.obstacleIDs[i]].pos[1]

        if end - begin > self.max_leaf_size:  # no leaf node      max_leaf_size=10
            vertical = (self.obstacleTree[node].maxX - self.obstacleTree[node].minX) > (
                    self.obstacleTree[node].maxY - self.obstacleTree[node].minY)  # vertical split
            splitValue = 0.5 * (self.obstacleTree[node].maxX + self.obstacleTree[node].minX) if vertical else 0.5 * (
                    self.obstacleTree[node].maxY + self.obstacleTree[node].minY)

            l = begin  # left
            r = end - 1  # right
            while True:
                while (l <= r) and ((self.obstacles[self.obstacleIDs[l]].pos[0]
                if vertical else self.obstacles[self.obstacleIDs[l]].pos[1]) < splitValue):
                    l += 1
                while (r >= l) and ((self.obstacles[self.obstacleIDs[r]].pos[0]
                if vertical else self.obstacles[self.obstacleIDs[r]].pos[1]) >= splitValue):
                    r -= 1
                if l > r:
                    break
                else:
                    self.obstacleIDs[l], self.obstacleIDs[r] = self.obstacleIDs[r], self.obstacleIDs[l]
                    l += 1
                    r -= 1

            leftsize = l - begin

            if leftsize == 0:
                leftsize += 1
                l += 1
                r += 1

            self.obstacleTree[node].left = node + 1
            self.obstacleTree[node].right = node + 1 + (2 * leftsize - 1)

            self.buildObstacleTreeRecursive(begin, l, self.obstacleTree[node].left)
            self.buildObstacleTreeRecursive(l, end, self.obstacleTree[node].right)

    def buildObstacleTree(self):
        if self.obstacleIDs:
            self.buildObstacleTreeRecursive(0, len(self.obstacleIDs), 0)

    def queryObstacleTreeRecursive(self, agent, rangeSq, node):
        if self.obstacleTree:
            if self.obstacleTree[node].end - self.obstacleTree[node].begin <= self.max_leaf_size:
                for i in range(self.obstacleTree[node].begin, self.obstacleTree[node].end):
                    agent.insertObstacleNeighbor(self.obstacles[i], rangeSq)
            else:
                distSqLeft = 0.0
                distSqRight = 0.0
                if agent.pos_global_frame[0] < self.obstacleTree[self.obstacleTree[node].left].minX:
                    distSqLeft += sqr(self.obstacleTree[self.obstacleTree[node].left].minX - agent.pos_global_frame[0])
                elif agent.pos_global_frame[0] > self.obstacleTree[self.obstacleTree[node].left].maxX:
                    distSqLeft += sqr(agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].left].maxX)
                if agent.pos_global_frame[1] < self.obstacleTree[self.obstacleTree[node].left].minY:
                    distSqLeft += sqr(self.obstacleTree[self.obstacleTree[node].left].minY - agent.pos_global_frame[1])
                elif agent.pos_global_frame[1] > self.obstacleTree[self.obstacleTree[node].left].maxY:
                    distSqLeft += sqr(agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].left].maxY)
                if agent.pos_global_frame[0] < self.obstacleTree[self.obstacleTree[node].right].minX:
                    distSqRight += sqr(self.obstacleTree[self.obstacleTree[node].right].minX - agent.pos_global_frame[0])
                elif agent.pos_global_frame[0] > self.obstacleTree[self.obstacleTree[node].right].maxX:
                    distSqRight += sqr(agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].right].maxX)
                if agent.pos_global_frame[1] < self.obstacleTree[self.obstacleTree[node].right].minY:
                    distSqRight += sqr(self.obstacleTree[self.obstacleTree[node].right].minY - agent.pos_global_frame[1])
                elif agent.pos_global_frame[1] > self.obstacleTree[self.obstacleTree[node].right].maxY:
                    distSqRight += sqr(agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].right].maxY)

                if distSqLeft < distSqRight:
                    if distSqLeft < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)
                    if distSqRight < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)
                else:
                    if distSqRight < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)
                    if distSqLeft < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)

    def computeObstacleNeighbors(self, agent, rangeSq):
        self.queryObstacleTreeRecursive(agent, rangeSq, 0)

    def queryVisibilityRecursive(self, q1, q2, radius, node):
        if node.obstacleID == -1:
            return True
        else:
            obstacle = self.obstacles[node.obstacleID]

            q1_leftof_i = leftOf(obstacle.p1, obstacle.p2, q1)
            q2_leftof_i = leftOf(obstacle.p1, obstacle.p2, q2)

            if q1_leftof_i >= 0 and q2_leftof_i >= 0:
                return self.queryVisibilityRecursive(q1, q2, radius, node.left)
            elif q1_leftof_i <= 0 and q2_leftof_i <= 0:
                return self.queryVisibilityRecursive(q1, q2, radius, node.right)
            else:
                p1_leftof_q = leftOf(q1, q2, obstacle.p1)
                p2_leftof_q = leftOf(q1, q2, obstacle.p2)
                invLength_q = 1.0 / l2norm(q2, q1) ** 2

                return (p1_leftof_q * p2_leftof_q >= 0
                        and sqr(p1_leftof_q) * invLength_q >= sqr(radius)
                        and sqr(p2_leftof_q) * invLength_q >= sqr(radius)
                        and self.queryVisibilityRecursive(q1, q2, radius, node.left)
                        and self.queryVisibilityRecursive(q1, q2, radius, node.right))

