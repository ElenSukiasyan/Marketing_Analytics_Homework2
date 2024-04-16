from bandit import Bandit, EpsilonGreedy, ThompsonSampling, Visualization, comparison
Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000

eg_results = EpsilonGreedy(Bandit).experiment(Bandit_Reward, NumberOfTrials)
EpsilonGreedy(Bandit).report(NumberOfTrials, eg_results, algorithm="EpsilonGreedy")
Visualization().plot1(NumberOfTrials, eg_results, 'Epsilon Greedy')

ts_results = ThompsonSampling(Bandit).experiment(Bandit_Reward, NumberOfTrials)
Visualization().plot1(NumberOfTrials, ts_results, algorithm='Thompson Sampling')

Visualization().plot2(eg_results, ts_results)
comparison(NumberOfTrials, eg_results, ts_results)