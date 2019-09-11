import os
import sys
import numpy as np
import math

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
				val = data[s*self.numStates + a].split('\t')
				for sPrime in range(0,self.numStates):
					self.rewards[s][a][sPrime] = float(val[sPrime])

		data = data[self.numStates*self.numActions:]
		self.transitions = np.zeros([self.numStates,self.numActions,self.numStates])
		for s in range(0,self.numStates):
			for a in range(0,self.numActions):
				val = data[s*self.numStates + a].split('\t')
				for sPrime in range(0,self.numStates):
					self.transitions[s][a][sPrime] = float(val[sPrime])

		data = data[self.numStates*self.numActions:]
		self.discount = float(data[0].split()[0])
		self.type = data[1].split()[0]


if __name__ == '__main__':
	# Read input parameters
	for i in range(len(sys.argv)):
		if (sys.argv[i] == "--mdp"):
			path = sys.argv[i+1]
		elif (sys.argv[i] == "--algorithm"):
			algorithm = sys.argv[i+1]
		
	# Error handling based on input
	if not os.path.isfile(path): 
		print(path + " : No such file exists")
		exit(0)

	if algorithm not in ["lp","hpi"]:
		print(algorithm + " : No such algorithm exists")
		exit(0)


	# Load and Create MDP
	try: mdp = MDP(path)
	except: 
		print("Unable to load mdp, check mdp format")
		exit(0)



