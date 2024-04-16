A/B Testing
Scenario
You have four advertisement options (bandits), and your task is to design an experiment using
Epsilon Greedy and Thompson Sampling.
Design of Experiment
A bandit class has already been created for you. It is an abstract class with abstract methods. You
must not exclude anything from the Bandit() class. However, you can add more stuff if you
need.
Bandit_Reward=[1,2,3,4]
NumberOfTrials: 20000
1. Create a Bandit Class
2. Create EpsilonGreedy() and ThompsonSampling() classes and methods (inherited
from Bandit()).
1. Epsilon-greedy:
1. decay epsilon by 1/t
2. design the experiment
2. Thompson Sampling
1. design with known precision
2. design the experiment
3. Report:
1. Visualize the learning process for each algorithm (plot1())
2. Visualize cumulative rewards from E-Greedy and Thompson Sampling.
3. Store the rewards in a CSV file ({Bandit, Reward, Algorithm})
4. Print cumulative reward
5. Print cumulative regret
Note the values of epsilon and precision are up to you to decide.
Submission
1. The code must be well documented; I’d recommend using pyment package or chatgpt
2. requirments.txt
3. We will not continue checking after the error message (regardless of the error).
4. Late submissions will be treated according to the rules written in the syllabus.
5. Push the codes to GitHub and submit only the link of a repo to Moodle
BONUS
Suggest better implementation plan (10 points )