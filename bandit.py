from abc import ABC, abstractmethod
from logs import *
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0 #estimate of average reward
        self.N = 0
        self.r_estimate = 0 #estimate of average regret

    def __repr__(self):
        return f'Arm - Win Rate: {self.p}'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    def report(self, N, results, algorithm):
        cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        
        # Save experiment data to a CSV file
        data_exp = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_exp.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results to a CSV file
        data_csv = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_csv.to_csv(f'{algorithm}_rewards.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p}')
            print(f'Pulled {bandits[b].N} times')
            print(f'Estimated average reward: {round(bandits[b].p_estimate, 4)}')
            print(f'Estimated average regret: {round(bandits[b].r_estimate, 4)}')
            print() 
                    
        print(f"Cumulative Reward : {sum(reward)}")
        print(f"Cumulative Regret : {cumulative_regret[-1]}")

        if algorithm == 'EpsilonGreedy':                            
                percent_suboptimal = round((count_suboptimal / N) * 100, 4)
                print("Percent suboptimal: {:.4f}%".format(percent_suboptimal))

class Visualization:
    def plot1(self, N, results, algorithm='EpsilonGreedy'):        
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        #linear
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        #log
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()


    def plot2(self, results_eg, results_ts):
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):

    def __init__(self, p):
        super().__init__(p)

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate


    def experiment(self, BANDIT_REWARDS, N, t=1):
        epsilon = 1 / t
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)  
        count_suboptimal = 0
        rewards = np.empty(N)
        chosen_bandits = np.empty(N)

        for i in range(N):
            if np.random.random() < epsilon:
                chosen_bandit_idx = np.random.choice(len(bandits))
            else:
                chosen_bandit_idx = np.argmax([bandit.p_estimate for bandit in bandits])

            reward = bandits[chosen_bandit_idx].pull()
            bandits[chosen_bandit_idx].update(reward)
            if chosen_bandit_idx != true_best:
                count_suboptimal += 1

            rewards[i] = reward
            chosen_bandits[i] = chosen_bandit_idx

            # Update epsilon for next iteration
            t += 1
            epsilon = 1 / t

        cumulative_reward_average = np.cumsum(rewards) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(rewards)
        cumulative_regret = np.maximum(N * max(means) - cumulative_reward, 0)

        return cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandits, rewards, count_suboptimal


class ThompsonSampling(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau_ = 1

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau_) + self.p
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        self.p_estimate = (self.tau_ * x + self.lambda_ * self.p_estimate) / (self.tau_ + self.lambda_)
        self.lambda_ += self.tau_
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        x = np.linspace(-3, 6, 100)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("{} trials of Bandit Distribution".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        bandits = [ThompsonSampling(i) for i in BANDIT_REWARDS]
        sample_points = [5, 50, 500, 5000]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()
            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max([b.p for b in bandits]) - cumulative_reward[i]
        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward 


def comparison(N, results_eg, results_ts):
    cumulative_reward_average_eg = results_eg[0]
    cumulative_reward_average_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    plt.figure(figsize=(12, 6))

    cumulative_rewards_eg = np.cumsum(reward_eg)
    cumulative_rewards_ts = np.cumsum(reward_ts)

    cumulative_regret_eg = np.cumsum(regret_eg)
    cumulative_regret_ts = np.cumsum(regret_ts)

    # Plot stacked area chart for cumulative rewards
    plt.subplot(1, 2, 1)
    plt.stackplot(range(N), cumulative_rewards_eg, cumulative_rewards_ts, labels=['Epsilon Greedy', 'Thompson Sampling'], colors=['blue', 'green'])
    plt.title('Cumulative Rewards Comparison')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Rewards')
    plt.legend()

    # Plot the optimal reward line
    optimal_reward = max([b.p for b in bandits_eg])
    plt.axhline(y=optimal_reward * N, color='red', linestyle='--', label='Optimal Reward')
    plt.legend()

    # Plot the cumulative average rewards
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_average_eg, label='Epsilon Greedy', color='blue')
    plt.plot(cumulative_reward_average_ts, label='Thompson Sampling', color='green')
    plt.axhline(y=optimal_reward, color='red', linestyle='--', label='Optimal Reward')
    plt.title('Cumulative Average Rewards Comparison')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the cumulative average rewards
    plt.subplot(1, 2, 2)
    plt.plot(results_eg[0], label='Epsilon Greedy', color='blue')
    plt.plot(results_ts[0], label='Thompson Sampling', color='green')
    plt.axhline(y=max([b.p for b in results_eg[3]]), color='red', linestyle='--', label='Optimal Reward')
    plt.title('Cumulative Average Rewards Comparison')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()