import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

class BEKEnv(gym.Env):
    def __init__(self, area_min=0, area_max=200000, goal_size=2000, T0=100000, dt=1.0, max_steps=1000000):
        super(BEKEnv, self).__init__()
        self.area_min = area_min
        self.area_max = area_max
        self.goal_size = goal_size
        self.T0 = T0
        self.dt = dt
        self.max_steps = max_steps

        self.max_speed = 10.0
        self.max_acc = 1.0
        self.max_delta_angle = np.pi / 8

        low_state = np.array([area_min, area_min, 0, -np.pi, 0], dtype=np.float32)
        high_state = np.array([area_max, area_max, self.max_speed, np.pi, 1e4], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_acc, -self.max_delta_angle], dtype=np.float32),
            high=np.array([self.max_acc, self.max_delta_angle], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.start = np.random.uniform(self.area_min, 50000, size=2)
        self.goal_corner = np.random.uniform(150000, self.area_max - self.goal_size, size=2)
        self.goal = np.array([
            [self.goal_corner[0], self.goal_corner[1]],
            [self.goal_corner[0] + self.goal_size, self.goal_corner[1] + self.goal_size]
        ])
        self.pos = self.start.copy()
        self.v = 0.0
        self.angle = np.random.uniform(-np.pi, np.pi)
        self.t = 0.0
        self.steps = 0
        self.trajectory = [self.pos.copy()]
        return self._get_state()

    def _get_state(self):
        return np.array([*self.pos, self.v, self.angle, self.t], dtype=np.float32)

    def _in_goal(self, pos):
        return (self.goal[0,0] <= pos[0] <= self.goal[1,0]) and (self.goal[0,1] <= pos[1] <= self.goal[1,1])

    def step(self, action):
        acc = float(np.clip(action[0], -self.max_acc, self.max_acc))
        delta_angle = float(np.clip(action[1], -self.max_delta_angle, self.max_delta_angle))

        self.angle += delta_angle
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

        self.v = np.clip(self.v + acc * self.dt, 0, self.max_speed)
        self.pos[0] += self.v * np.cos(self.angle) * self.dt
        self.pos[1] += self.v * np.sin(self.angle) * self.dt
        self.t += self.dt
        self.steps += 1
        self.trajectory.append(self.pos.copy())

        done = self._in_goal(self.pos) or self.steps >= self.max_steps
        reward = 0.0
        if done:
            if self._in_goal(self.pos):
                reward = np.exp(self.T0 - self.t)
            else:
                reward = -1.0
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        traj = np.array(self.trajectory)
        plt.figure(figsize=(8, 8))
        plt.plot(traj[:, 0], traj[:, 1], color='blue', label='Траектория')
        plt.scatter([self.start[0]], [self.start[1]], color='green', s=100, label='Старт')
        plt.gca().add_patch(
            plt.Rectangle(self.goal_corner, self.goal_size, self.goal_size, color='red', alpha=0.3, label='Целевая область')
        )
        plt.title('Оптимальная траектория движения БЭК')
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.xlim(self.area_min, self.area_max)
        plt.ylim(self.area_min, self.area_max)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()


env = BEKEnv()
state = env.reset()
done = False

while not done:
    # Здесь можно использовать RL-агента, а для примера — случайные действия:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)

env.render()
print(f"Reward: {reward:.3f}, Time: {env.t:.1f} сек")
