import time
import numpy as np
from math import sqrt, sin, cos, atan2, asin, pi, floor, acos, asin
from itertools import combinations, product

from mamp.envs import Config
from mamp.util import absSq, l2norm, norm, sqr, det, mod2pi, pi_2_pi, normalize, angle_2_vectors
from mamp.policies.policy import Policy


class Line(object):
    def __init__(self):
        self.direction = np.array([0.0, 1.0])  # The direction of the directed line.
        self.point = np.array([0.0, 0.0])  # A point on the directed line.


class ORCAPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="ORCAPolicy")
        self.type = "internal"
        self.now_goal = None
        self.epsilon = 1e-5
        self.orcaLines = []
        self.newVelocity = np.array([0.0, 0.0])

    def find_next_action(self, obs, dict_comm, agent, kdTree):
        ts = time.time()
        self.now_goal = agent.goal_global_frame
        self.orcaLines.clear()
        self.newVelocity = np.array([0.0, 0.0])
        invTimeHorizonObst = 1.0 / agent.timeHorizonObst
        invTimeHorizon = 1.0 / agent.timeHorizon
        v_pref = compute_v_pref(agent)
        computeNeighbors(agent, kdTree)

        for obj in agent.neighbors:
            obj = obj[0]
            relativePosition = obj.pos_global_frame - agent.pos_global_frame
            relativeVelocity = agent.vel_global_frame - obj.vel_global_frame
            distSq = absSq(relativePosition)
            agent_rad = agent.radius + 0.05
            obj_rad = obj.radius + 0.05
            combinedRadius = agent_rad + obj_rad
            combinedRadiusSq = sqr(combinedRadius)

            line = Line()

            if distSq > combinedRadiusSq:   # No collision.
                w = relativeVelocity - invTimeHorizon * relativePosition
                # Vector from cutoff center to relative velocity.
                wLengthSq = absSq(w)

                dotProduct1 = np.dot(w, relativePosition)

                if dotProduct1 < 0.0 and sqr(dotProduct1) > combinedRadiusSq * wLengthSq:
                    # Project on cut-off circle.
                    wLength = sqrt(wLengthSq)
                    unitW = w / wLength

                    line.direction = np.array([unitW[1], -unitW[0]])
                    u = (combinedRadius * invTimeHorizon - wLength) * unitW
                else:
                    # Project on legs.
                    leg = sqrt(distSq - combinedRadiusSq)

                    if det(relativePosition, w) > 0.0 > 0.0:
                        # Project on left leg.
                        x = relativePosition[0] * leg - relativePosition[1] * combinedRadius
                        y = relativePosition[0] * combinedRadius + relativePosition[1] * leg
                        line.direction = np.array([x, y]) / distSq
                    else:
                        # Project on right leg.
                        x = relativePosition[0] * leg + relativePosition[1] * combinedRadius
                        y = -relativePosition[0] * combinedRadius + relativePosition[1] * leg
                        line.direction = -np.array([x, y]) / distSq

                    dotProduct2 = np.dot(relativeVelocity, line.direction)

                    u = dotProduct2 * line.direction - relativeVelocity
            else:
                # Collision.Project on cut - off circle of time timeStep.
                invTimeStep = 1.0 / agent.timeStep

                # Vector from cutoff center to relative velocity.
                w = relativeVelocity - invTimeStep * relativePosition
                wLength = norm(w)
                unitW = w / wLength

                line.direction = np.array([unitW[1], -unitW[0]])
                u = (combinedRadius * invTimeStep - wLength) * unitW

            line.point = agent.vel_global_frame + 0.5 * u
            self.orcaLines.append(line)

        lineFail = self.linearProgram2(self.orcaLines, agent.maxSpeed, v_pref, False)

        if lineFail < len(self.orcaLines):
            self.linearProgram3(self.orcaLines, 0, lineFail, agent.maxSpeed)
            # print(222, self.newVelocity)

        vA_post = np.array([self.newVelocity[0], self.newVelocity[1]])
        vA = agent.vel_global_frame
        te = time.time()
        cost_step = te - ts
        agent.total_time += cost_step
        action = to_unicycle(vA_post, agent)
        agent.vel_global_unicycle[0] = round(1.0 * action[0], 5)
        agent.vel_global_unicycle[1] = round(0.22 * 0.5 * action[1] / Config.DT, 5)
        theta = angle_2_vectors(vA, vA_post)
        dist = l2norm(agent.pos_global_frame, agent.goal_global_frame)
        if theta > agent.max_heading_change:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist, 'theta:', theta)
        else:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist)

        return action

    def linearProgram1(self, lines, lineNo, radius, optVelocity, directionOpt):
        """
           Solves a one-dimensional linear program on a specified line subject to linear
           constraints defined by lines and a circular constraint.
           Args:
               lines (list): Lines defining the linear constraints.
               lineNo (int): The specified line constraint.
               radius (float): The radius of the circular constraint.
               optVelocity (Vector2): The optimization velocity.
               directionOpt (bool): True if the direction should be optimized.
           Returns:
               bool: True if successful.
               Vector2: A reference to the result of the linear program.
        """
        dotProduct = np.dot(lines[lineNo].point, lines[lineNo].direction)
        discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines[lineNo].point)

        if discriminant < 0.0:
            # Max speed circle fully invalidates line lineNo.
            return False

        sqrtDiscriminant = sqrt(discriminant)
        tLeft = -dotProduct - sqrtDiscriminant
        tRight = -dotProduct + sqrtDiscriminant

        for i in range(lineNo):
            denominator = det(lines[lineNo].direction, lines[i].direction)
            numerator = det(lines[i].direction, lines[lineNo].point - lines[i].point)

            if abs(denominator) <= self.epsilon:
                # Lines lineNo and i are (almost) parallel.
                if numerator < 0.0:
                    return False
                else:
                    continue

            t = numerator / denominator

            if denominator >= 0.0:
                # Line i bounds line lineNo on the right.
                tRight = min(tRight, t)
            else:
                # * Line i bounds line lineNo on the left.
                tLeft = max(tLeft, t)

            if tLeft > tRight:
                return False

        if directionOpt:
            # Optimize direction.
            if np.dot(optVelocity, lines[lineNo].direction) > 0.0:
                # Take right extreme.
                self.newVelocity = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                # Take left extreme.
                self.newVelocity = lines[lineNo].point + tLeft * lines[lineNo].direction
        else:
            # Optimize closest point.
            t = np.dot(lines[lineNo].direction, optVelocity - lines[lineNo].point)

            if t < tLeft:
                self.newVelocity = lines[lineNo].point + tLeft * lines[lineNo].direction
            elif t > tRight:
                self.newVelocity = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                self.newVelocity = lines[lineNo].point + t * lines[lineNo].direction

        return True

    def linearProgram2(self, lines, radius, optVelocity, directionOpt):
        """
           Solves a two-dimensional linear program subject to linear constraints defined by
           lines and a circular constraint.
           Args:
               lines (list): Lines defining the linear constraints.
               radius (float): The radius of the circular constraint.
               optVelocity (Vector2): The optimization velocity.
               directionOpt (bool): True if the direction should be optimized.
           Returns:
               int: The number of the line it fails on, and the number of lines if successful.
               Vector2: A reference to the result of the linear program.
        """
        if directionOpt:
            # Optimize direction. Note that the optimization velocity is of unit length in this case.
            self.newVelocity = optVelocity * radius
        elif absSq(optVelocity) > sqr(radius):
            # Optimize closest point and outside circle.
            self.newVelocity = normalize(optVelocity) * radius
        else:
            # Optimize closest point and inside circle.
            self.newVelocity = optVelocity
        for i in range(len(lines)):
            if det(lines[i].direction, lines[i].point - self.newVelocity) > 0.0:
                # Result does not satisfy constraint i.Compute new optimal result.
                tempResult = self.newVelocity

                if not self.linearProgram1(lines, i, radius, optVelocity, directionOpt):
                    self.newVelocity = tempResult
                    return i

        return len(lines)

    def linearProgram3(self, lines, numObstLines, beginLine, radius):
        """
        Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular constraint.
        Args:
            lines (list): Lines defining the linear constraints.
            numObstLines (int): Count of obstacle lines.
            beginLine (int): The line on which the 2-d linear program failed.
            radius (float): The radius of the circular constraint.
        """
        distance = 0.0

        for i in range(beginLine, len(lines)):
            if det(lines[i].direction, lines[i].point - self.newVelocity) > distance:
                # Result does not satisfy constraint of line i.
                projLines = []
                for j in range(numObstLines):
                    projLines.append(lines[j])

                for j in range(numObstLines, i):
                    line = Line()
                    determinant = det(lines[i].direction, lines[j].direction)

                    if abs(determinant) <= self.epsilon:
                        # Line i and line j are parallel.
                        if np.dot(lines[i].direction, lines[j].direction) > 0.0:
                            # Line i and line j point in the same direction.
                            continue
                        else:
                            # Line i and line j point in opposite direction.
                            line.point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        p1 = (det(lines[j].direction, lines[i].point - lines[j].point) / determinant)
                        line.point = lines[i].point + p1 * lines[i].direction

                    line.direction = normalize(lines[j].direction - lines[i].direction)
                    projLines.append(line)

                tempResult = self.newVelocity
                optVelocity = np.array([-lines[i].direction[1], lines[i].direction[0]])
                if self.linearProgram2(projLines, radius, optVelocity, True) < len(projLines):
                    """
                    This should in principle not happen. The result is by definition already 
                    in the feasible region of this linear program. If it fails, it is due to 
                    small floating point error, and the current result is kept.
                    """
                    self.newVelocity = tempResult

                distance = det(lines[i].direction, lines[i].point - self.newVelocity)


def compute_v_pref(agent):
    goal = agent.goal_global_frame
    diff = goal - agent.pos_global_frame
    v_pref = agent.pref_speed * normalize(diff)
    if l2norm(goal, agent.pos_global_frame) < Config.NEAR_GOAL_THRESHOLD:
        v_pref = np.zeros_like(v_pref)
    return v_pref


def computeNeighbors(agent, kdTree):
    agent.neighbors.clear()

    # check obstacle neighbors
    rangeSq = agent.neighborDist ** 2
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = 1.0 * agent.neighborDist ** 2
    kdTree.computeObstacleNeighbors(agent, rangeSq)

    if agent.in_collision:
        return

    # check other agents
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = agent.neighborDist ** 2
    kdTree.computeAgentNeighbors(agent, rangeSq)


def to_unicycle(vA_post, agent):
    vA_post = np.array(vA_post)
    norm_vA = norm(vA_post)
    yaw_next = mod2pi(atan2(vA_post[1], vA_post[0]))
    yaw_current = mod2pi(agent.heading_global_frame)
    delta_theta = yaw_next - yaw_current
    delta_theta = pi_2_pi(delta_theta)
    if delta_theta < -pi:
        delta_theta = delta_theta + 2 * pi
    if delta_theta > pi:
        delta_theta = delta_theta - 2 * pi
    if delta_theta >= 1.0:
        delta_theta = 1.0
    if delta_theta <= -1:
        delta_theta = -1.0
    action = np.array([norm_vA, delta_theta])
    return action

