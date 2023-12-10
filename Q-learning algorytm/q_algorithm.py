import numpy as np
import gymnasium as gym
import pygame


class QLearningSolver:
    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((self.observation_space, self.action_space))

    def __call__(self, state: int, action: int) -> float:
        return self.q_table[state, action]

    def update(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.q_table[state, action] = reward

    def get_best_action_index(self, state: int) -> int:
        return np.argmax(self.q_table[state, :])

    def get_best_reward(self, state: int) -> float:
        return max(self.q_table[state, :])

    def __repr__(self):
        info = f"QLearningSolver: observation_space: {self.observation_space}, \
            action_space: {self.action_space}\n"
        info2 = f"Learning rate: {self.learning_rate}, gamma: {self.gamma}, epsilon: {self.epsilon}\n"
        return info + info2 + f"Qtable: \n{self.q_table}"

    def __str__(self):
        return self.__repr__()


def get_trained_q_solver_for_taxi_problem(n_episdoes: int, max_iter_per_episode: int) -> QLearningSolver:
    q_solver = QLearningSolver(500, 6)

    env = gym.make('Taxi-v3')
    for _ in range(n_episdoes):
        starting_state = env.reset()
        curr_state = starting_state[0]

        if_finished = None
        iters = 0

        while not if_finished and iters <= max_iter_per_episode:
            if np.random.uniform(0, 1) < q_solver.epsilon:
                action = env.action_space.sample()
            else:
                action = q_solver.get_best_action_index(curr_state)

            next_state, reward, if_finished, _, _ = env.step(action)
            lr = q_solver.learning_rate
            new_reward = (1 - lr) * q_solver(curr_state, action) + lr * \
                (reward + q_solver.gamma * q_solver.get_best_reward(next_state))
            q_solver.update(curr_state, action, new_reward)
            curr_state = next_state
            iters += 1

    env.close()
    return q_solver


def show_visualization(*, train_n_episodes: int = 2500, train_max_iter_per_episode: int = 300,
                       max_steps_in_episode: int = 30, ms_delay: int = 100,
                       if_verbose: bool = False) -> None:
    env = gym.make('Taxi-v3', render_mode="human",
                   max_episode_steps=max_steps_in_episode)
    starting_state = env.reset()
    curr_state = starting_state[0]
    q_solver = get_trained_q_solver_for_taxi_problem(train_n_episodes,
                                                     train_max_iter_per_episode)

    if_finished = None
    if_truncated = None
    info = None
    while not if_finished and not if_truncated:
        env.render()
        if not info:
            action = np.random.choice((np.where(starting_state[1]['action_mask'] == 1))[0])
        else:
            action = q_solver.get_best_action_index(curr_state)
            if if_verbose:
                print(f"Best action is: {action}")

        next_state, _, if_finished, if_truncated, info = env.step(action)
        curr_state = next_state

        if if_verbose:
            print("Aktualnie można wykonać akcje: " + str(info['action_mask']))
            print(q_solver)
        pygame.time.delay(ms_delay)

    env.render()
    env.close()


if __name__ == "__main__":
    show_visualization()
