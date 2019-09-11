import os
import sys
import numpy as np
import math

class MultiArmedBandit:
	def __init__(self,path,randomSeed):
		file = open(path,'r')
		lines = file.readlines()
		self.variants = len(lines)
		self.payouts = [float(line.strip()) for line in lines]
		np.random.seed(randomSeed)

	def pullArm(self,k):
		r = np.random.uniform(0,1)
		if r < self.payouts[k]: return 1
		else: return 0

	def roundRobin(self,horizon):
		k = 0
		cumulativeReward = 0
		for turn in range(horizon):
			# Pull an arm and update the cumulative reward
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome
			k = (k + 1) % self.variants 

		maxReward = max(self.payouts) * horizon
		regret = maxReward - cumulativeReward
		return regret

	def epsilonGreedy(self,epsilon,horizon):
		cumulativeReward = 0

		empericalPayouts = [0] * self.variants
		rewards = [0] * self.variants
		pulls = [0] * self.variants

		# Estimating emperical payouts for n learning trials
		k = 0
		n = min(self.variants,horizon)
		for turn in range(n):
			# Pull kth arm and update the cumulative reward
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome 
			k = (k + 1) % self.variants 

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]


		# Explore / exploit with coresponding probabilites, for remaining turns 
		for turn in range(n,horizon):
			# Choose whether to explore or exploit
			r = np.random.uniform(0,1)
			if (r < epsilon): 
				# Explore : Choose and arm at random
				k = np.random.randint(self.variants)
			else:
				# Exploit : Choose the arm with maximum emperical payout 
				k = empericalPayouts.index(max(empericalPayouts))

			# Pull kth arm and update the cumulative reward
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]

		maxReward = max(self.payouts) * horizon
		regret = maxReward - cumulativeReward
		return regret

	def ucb(self,horizon):
		cumulativeReward = 0

		empericalPayouts = [0] * self.variants
		rewards = [0] * self.variants
		pulls = [0] * self.variants

		# Estimating emperical payouts for n learning trials
		k = 0
		n = min(self.variants,horizon)
		for turn in range(n):
			# Pull kth arm and update the cumulative reward
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome 
			k = (k + 1) % self.variants 

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]


		# Define upper confidence bound and use it for sampling
		ucbValues = [0] * self.variants
		for turn in range(n,horizon):
			for i in range(self.variants):
				term = math.sqrt(2 * math.log(turn) / pulls[i])
				ucbValues[i] = empericalPayouts[i] + term
			# Choose the arm with maximum ucb value and pull it
			k = ucbValues.index(max(ucbValues))
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]

		maxReward = max(self.payouts) * horizon
		regret = maxReward - cumulativeReward
		return regret

	def klDivergence(self, p,q):
		result = 0
		if p > 0: result = result + p * (math.log(p) - math.log(q))
		if p < 1: result = result + (1-p) * (math.log(1-p) - math.log(1-q))
		return result

	def solve(self, pHat ,upperBound):
		# Define maxIter and precision
		maxIter = 25
		precision = 1e-6

		# Use binary search to find the optimum solution
		l = pHat
		r = 1
		for i in range(maxIter):
			m = (l + r) / 2
			kl = self.klDivergence(pHat,m)
			if (abs(kl - upperBound) < precision): break
			if (kl <= upperBound): l = m
			else: r = m
		return m

	def generateKlUcbValues(self, empericalPayouts, turn, pulls):
		klUcbValues = [0] * self.variants
		for i in range(self.variants):
			# Get value using the below formula
			upperBound = (math.log(turn) + 3 * math.log(math.log(turn))) / pulls[i]
			klUcbValues[i] = self.solve(empericalPayouts[i],upperBound)
		return klUcbValues
		
	def klUcb(self,horizon):
		cumulativeReward = 0

		empericalPayouts = [0] * self.variants
		rewards = [0] * self.variants
		pulls = [0] * self.variants

		# Estimating emperical payouts for n learning trials
		k = 0
		n = min(self.variants,horizon)
		for turn in range(n):
			# Pull kth arm and update the cumulative reward
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome 
			k = (k + 1) % self.variants 

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]


		# Define KL- upper confidence bound and use it for sampling
		klUcbValues = [0] * self.variants
		for turn in range(n,horizon):
			klUcbValues = self.generateKlUcbValues(empericalPayouts, turn, pulls)
			# Choose the arm with maximum KL ucb value and pull it
			k = klUcbValues.index(max(klUcbValues))
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome

			# Update kth arms reward and pull count
			rewards[k] = rewards[k] + outcome
			pulls[k] = pulls[k] + 1
			empericalPayouts[k] = rewards[k]/pulls[k]

		maxReward = max(self.payouts) * horizon
		regret = maxReward - cumulativeReward
		return regret


	def thompsonSampling(self,horizon):
		cumulativeReward = 0
		successes = [0] * self.variants
		faliures = [0] * self.variants
		for turn in range(horizon):
			# Calculate xa values by drawing from a beta distribution
			x = [0] * self.variants
			for i in range(self.variants):
				x[i] = np.random.beta(successes[i]+1,faliures[i]+1)
			# Choose the arm with maximum x value
			k = x.index(max(x))
			outcome = self.pullArm(k)
			cumulativeReward = cumulativeReward + outcome

			# Update successes and faliures
			successes[k] = successes[k] + outcome
			faliures[k] = faliures[k] + int(not(outcome))

		maxReward = max(self.payouts) * horizon
		regret = maxReward - cumulativeReward
		return regret

if __name__ == '__main__':
	# Read input parameters
	for i in range(len(sys.argv)):
		if (sys.argv[i] == "--instance"):
			instance = sys.argv[i+1]
		elif (sys.argv[i] == "--algorithm"):
			algorithm = sys.argv[i+1]
		elif (sys.argv[i] == "--randomSeed"):
			randomSeed = sys.argv[i+1]
		elif (sys.argv[i] == "--epsilon"):
			epsilon = sys.argv[i+1]
		elif (sys.argv[i] == "--horizon"):
			horizon = sys.argv[i+1]

	# Error handling based on input
	if not os.path.isfile(instance): 
		print("Unable to find instance file")
		exit(0)

	if algorithm not in ["round-robin","epsilon-greedy","ucb","kl-ucb","thompson-sampling"]:
		print("Unable to find the algorithm")
		exit(0)


	# Handle types
	randomSeed = int(randomSeed)
	if (algorithm == "epsilon-greedy"): epsilon = float(epsilon)
	else: epsilon = "-"
	horizon = int(horizon)

	# Load and Create Instance
	try: 
		multiArmedBandit = MultiArmedBandit(instance,randomSeed)
	except: 
		print("Unable to load instance, check instance format")
		exit(0)


	# Calculate regret
	if algorithm == "round-robin":
		regret = multiArmedBandit.roundRobin(horizon)
	elif algorithm == "epsilon-greedy": 
		regret = multiArmedBandit.epsilonGreedy(epsilon,horizon)
	elif algorithm == "ucb": 
		regret = multiArmedBandit.ucb(horizon)
	elif algorithm == "kl-ucb": 
		regret = multiArmedBandit.klUcb(horizon)
	elif algorithm == "thompson-sampling": 
		regret = multiArmedBandit.thompsonSampling(horizon)

	# Print output in format - instance, algorithm, random seed, epsilon, horizon, REG
	output = instance + "," + algorithm + "," + str(randomSeed) + "," + str(epsilon) + "," + str(horizon) + "," + str(regret)
	print(output)

	# # Save output for report
	# file = open('output.txt','+a')
	# file.write(output + '\n')
	# file.close()


