import os
import sys
import numpy as np
import math
import pulp

class MDP:
	def __init__(self,path):
		# load and parse MDP file
		f = open(path)
		data = f.readlines()
		self.numStates = int(data[0].split()[0])
		self.numActions = int(data[1].split()[0])
		
		data = data[2:]
		self.rewards = np.zeros([self.numStates,self.numActions,self.numStates])
		for s in range(0,self.numStates):
			for a in range(0,self.numActions):
				val = data[s*self.numActions + a].split('\t')
				for sPrime in range(0,self.numStates):
					self.rewards[s][a][sPrime] = float(val[sPrime])

		data = data[self.numStates*self.numActions:]
		self.transitions = np.zeros([self.numStates,self.numActions,self.numStates])
		for s in range(0,self.numStates):
			for a in range(0,self.numActions):
				val = data[s*self.numActions + a].split('\t')
				for sPrime in range(0,self.numStates):
					self.transitions[s][a][sPrime] = float(val[sPrime])

		data = data[self.numStates*self.numActions:]
		self.discount = float(data[0].split()[0])
		self.type = data[1].split()[0]

	def lp(self):
		# create the LP object
		problem = pulp.LpProblem("MDP Solver", pulp.LpMinimize)

		# create decision variables
		values = []
		for i in range(self.numStates):
			variable = str("V(" + str(i) + ")")
			variable = pulp.LpVariable(str(variable),lowBound=None, upBound=None, cat='Continuous', e=None)
			values.append(variable)

		# create objective function
		total_values = ""
		for value in values:
			total_values += value

		problem += total_values

		# create constrains
		for s in range(self.numStates):
			for a in range(self.numActions):
				empiricalValue = ""
				for sPrime in range(self.numStates):
					empiricalValue += self.transitions[s][a][sPrime]*(self.rewards[s][a][sPrime] + self.discount*values[sPrime]) 
				problem += (values[s] >= empiricalValue)

		# now run optimization
		optimization_result = problem.solve(solvers.PULP_CBC_CMD(fracGap=1e-8))
		assert optimization_result == pulp.LpStatusOptimal

		# get policy
		policy = []
		for s in range(self.numStates):
			maxAction = -1
			maxEmpiricalValue = -float('inf')
			for a in range(self.numActions):
				empiricalValue = 0
				for sPrime in range(self.numStates):
					empiricalValue += self.transitions[s][a][sPrime]*(self.rewards[s][a][sPrime] + self.discount*problem.variables()[sPrime].varValue) 
				if empiricalValue > maxEmpiricalValue:
					maxEmpiricalValue = empiricalValue
					maxAction = a
			policy.append(maxAction)


		# print output
		for i, v in enumerate(problem.variables()):
			print(str(float(int(v.varValue*1e8))/1e8) + "\t" + str(policy[i]))

	def evaluatePolicy(self,policy):
		# start with a random value function
		valuesPrev = np.zeros(self.numStates)

		while(True):
			valuesCurr = np.zeros(self.numStates)
			delta = 0

			for s in range(self.numStates):
				actionValues = policy[s]

				for a in range(self.numActions):
					valueMax = 0
					for sPrime in range(self.numStates):
						valuesCurr[s] += actionValues[a]*self.transitions[s][a][sPrime]*(self.rewards[s][a][sPrime] + self.discount*valuesPrev[sPrime])

				delta = max(delta,abs(valuesCurr[s]-valuesPrev[s]))

			if (delta < 1e-20): break
			valuesPrev = valuesCurr

		return valuesCurr  


	def hpi(self):
		# start with a random policy
		policy = np.ones([self.numStates, self.numActions]) / self.numActions

		while(True):
			# evaluate the current policy
			values = self.evaluatePolicy(policy)
			policyStable = True
			for s in range(self.numStates):
				actionValues = [0] * self.numActions

				# perform lookahead
				for a in range(self.numActions):
					for sPrime in range(self.numStates):
						actionValues[a] += self.transitions[s][a][sPrime]*(self.rewards[s][a][sPrime] + self.discount*values[sPrime])


				bestAction = np.argmax(actionValues)
				chosenAction = np.argmax(policy[s])
				if (bestAction != chosenAction): 
					policyStable = False

				policy[s] = np.eye(self.numActions)[bestAction]

			if(policyStable): break

		# print output
		for i in range(self.numStates):
			print(str(float(int(values[i]*1e8))/1e8) + "\t" + str(np.argmax(policy[i])))


if __name__ == '__main__':
	# read input parameters
	for i in range(len(sys.argv)):
		if (sys.argv[i] == "--mdp"):
			path = sys.argv[i+1]
		elif (sys.argv[i] == "--algorithm"):
			algorithm = sys.argv[i+1]
		
	# error handling based on input
	if not os.path.isfile(path): 
		print(path + " : No such file exists")
		exit(0)

	if algorithm not in ["lp","hpi"]:
		print(algorithm + " : No such algorithm exists")
		exit(0)


	# load and create MDP
	try: mdp = MDP(path)
	except: 
		print("Unable to load mdp, check mdp format")
		exit(0)

	# solve problem
	if (algorithm == 'lp'): mdp.lp()
	else: mdp.hpi()


