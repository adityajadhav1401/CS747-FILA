#!/usr/bin/env python3
import subprocess
import os

if (os.path.exists('output.txt')): os.remove('output.txt')
instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
# algorithms = ["round-robin", "epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
algorithms = ["epsilon-greedy"]
epsilons = [0.002, 0.02, 0.2]
horizons = [50, 200, 800, 3200, 12800, 51200, 204800]
seeds = range(50)

count = 0
for instance in instances:
	for algorithm in algorithms:
		for seed in seeds:
			for horizon in horizons:
				if (algorithm == "epsilon-greedy"):
					for epsilon in epsilons:
						command = "./bandit.sh --instance " + instance + " --algorithm " + algorithm 
						command = command + " --randomSeed " + str(seed) + " --epsilon " + str(epsilon)
						command = command + " --horizon " + str(horizon)
						count += 1
						subprocess.call(command,shell=True)
				else:
					command = "./bandit.sh --instance " + instance + " --algorithm " + algorithm 
					command = command + " --randomSeed " + str(seed) + " --epsilon " + "-"
					command = command + " --horizon " + str(horizon)
					count += 1		
					subprocess.call(command,shell=True)


print(count)