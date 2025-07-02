import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import csv
import os
import math
import random

class BEKTrajectory:
    def __init__(self, start_pos=(50000, 50000), speed=7, acceleration=2.5):
        self.v = speed
        self.a_c = acceleration
        self.R = self.v ** 2 / self.a_c
        self.x_points = [start_pos[0]]
        self.y_points = [start_pos[1]]
        self.segments = []
        self.log_id = 1
        self.obj_id = "001"
        self.current_time = 0  # В секундах
        self.current_direction = 0  # Начальное направление (вправо по оси X)
        self.log_entries = []

    def _log_event(self, event_type, acc=None, v=None):
        time_delta = timedelta(seconds=self.current_time)
        hours, remainder = divmod(time_delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        if time_str == "00:00:00" and event_type == "UMSVStartNewAcceleration":
            return 0

        log_entrus = {
            'ID': self.log_id,
            'Time': time_str,
            'ObjID': self.obj_id,
            'EventType': event_type,
            'Acc': acc if acc is not None else 0,
            'V': self.v
        }
        log_entry = (f"{self.log_id}\t{time_str}\t{self.obj_id}\t{event_type}\t"
                     f"{acc if acc is not None else 0}\t{v if v is not None else 0}")
        print(log_entry)
        self.log_id += 1
        self.log_entries.append(log_entrus)

    def add_straight(self, duration, direction):
        self._log_event("UMSVStartNewAcceleration", None, self.v)

        dx = self.v * duration * np.cos(np.radians(direction))
        dy = self.v * duration * np.sin(np.radians(direction))
        new_x = self.x_points[-1] + dx
        new_y = self.y_points[-1] + dy

        self.x_points.append(new_x)
        self.y_points.append(new_y)
        self.segments.append(('straight', duration, direction))

        self.current_time += duration

    def add_turn(self, turn_angle, turn_direction, steps=100):
        turn_time = np.radians(turn_angle) * self.R / self.v

        center_x = self.x_points[-1]
        center_y = self.y_points[-1]

        if turn_direction == 'right':
            angle_range = np.linspace(0, -np.radians(turn_angle), steps)
            self._log_event("UMSVStartNewAcceleration", self.a_c, self.v)
            center_x += self.R * np.cos(np.radians(self.current_direction + 90))
            center_y += self.R * np.sin(np.radians(self.current_direction + 90))
        else:
            self._log_event("UMSVStartNewAcceleration", -self.a_c, self.v)
            angle_range = np.linspace(0, np.radians(turn_angle), steps)
            center_x += self.R * np.cos(np.radians(self.current_direction - 90))
            center_y += self.R * np.sin(np.radians(self.current_direction - 90))

        x_turn = center_x + self.R * np.sin(angle_range)
        y_turn = center_y - self.R * np.cos(angle_range)

        self.x_points.extend(x_turn)
        self.y_points.extend(y_turn)
        self.segments.append(('turn', turn_angle, turn_direction))

        if turn_direction == 'right':
            self.current_direction -= turn_angle
        else:
            self.current_direction += turn_angle

        self.current_time += turn_time

    def save_log_to_csv(self, filename='trajectory_log.csv'):
        fieldnames = ['ID', 'Time', 'ObjID', 'EventType', 'Acc', 'V']

        file_exists = os.path.isfile(filename)

        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for entry in self.log_entries:
                writer.writerow(entry)

        print(f"\nЛоги сохранены в файл: {filename}")

def reward_function_scaled(T_0, T_max):
    return math.exp(T_0 - T_max)

class BEKTrajectoryEnv:
    def __init__(self, start_pos=(50000, 50000), speed=7, acceleration=2.5, target_pos=(100000, 100000), max_time=18000):
        self.start_pos = start_pos
        self.speed = speed
        self.acceleration = acceleration
        self.target_pos = target_pos
        self.max_time = max_time
        self.reset()

    def reset(self):
        self.bek = BEKTrajectory(start_pos=self.start_pos, speed=self.speed, acceleration=self.acceleration)
        self.done = False
        return (self.bek.x_points[-1], self.bek.y_points[-1], self.bek.current_direction, self.bek.current_time)

    def step(self, action):
        if self.done:
            raise Exception("Episode is done. Reset environment.")

        if action[0] == 'straight':
            self.bek.add_straight(action[1], self.bek.current_direction)
        elif action[0] == 'turn':
            self.bek.add_turn(action[1], action[2])

        dx = self.target_pos[0] - self.bek.x_points[-1]
        dy = self.target_pos[1] - self.bek.y_points[-1]
        dist_to_target = (dx**2 + dy**2)**0.5

        if dist_to_target < 1000:  # 1 км до цели
            self.done = True

        reward = reward_function_scaled(self.bek.current_time, self.max_time)

        if self.bek.current_time > self.max_time:
            self.done = True
            reward -= 10  # штраф за превышение времени

        state = (self.bek.x_points[-1], self.bek.y_points[-1], self.bek.current_direction, self.bek.current_time)
        return state, reward, self.done

class SimpleRLAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = {}
        self.actions = [
            ('straight', 600),  # 10 минут прямого движения
            ('turn', random.randint(0, 180), 'right'),
            ('turn',  random.randint(0, 180), 'left')
        ]
        self.alpha = 0.1  # скорость обучения
        self.gamma = 0.9  # коэффициент дисконтирования
        self.epsilon = 0.2  # вероятность случайного действия

    def get_state_key(self, state):
        x, y, direction, time = state
        return (round(x, -3), round(y, -3), round(direction / 45) * 45, round(time, -2))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon or state_key not in self.q_table:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0 for a in self.actions}

        predict = self.q_table[state_key][action]
        target = reward + self.gamma * max(self.q_table[next_state_key].values())
        self.q_table[state_key][action] += self.alpha * (target - predict)

    def train(self, episodes=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f}")

    def test(self):
        state = self.env.reset()
        done = False
        trajectory = [(state[0], state[1])]

        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            trajectory.append((next_state[0], next_state[1]))
            state = next_state

        return trajectory

def plot_trajectory(trajectory):
    x_points, y_points = zip(*trajectory)
    plt.figure(figsize=(12, 10))
    plt.plot(x_points, y_points, 'b-', linewidth=2, label='Траектория RL агента')
    plt.scatter(x_points[0], y_points[0], color='green', s=100, label=f'Старт ({x_points[0]}, {y_points[0]})')
    plt.scatter(x_points[-1], y_points[-1], color='red', s=100, label=f'Финиш ({x_points[-1]:.1f}, {y_points[-1]:.1f})')
    plt.xlabel('Ось X (м)', fontsize=12)
    plt.ylabel('Ось Y (м)', fontsize=12)
    plt.title('Траектория движения RL агента', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 200000)
    plt.ylim(0, 200000)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env = BEKTrajectoryEnv(target_pos=(100000, 100000), max_time=18000)
    agent = SimpleRLAgent(env)

    print("Начинаем обучение RL агента...")
    agent.train(episodes=100)

    print("Тестируем обученного агента...")
    trajectory = agent.test()

    plot_trajectory(trajectory)

    env.bek.save_log_to_csv('rl_trajectory_log.csv')
