import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt

class Gridword:
	def __init__(self, actionType, stochasticType):
		self.numRows = 7
		self.numCols = 10
		if (actionType == 0): self.numActions = 4
		else: self.numActions = 8

		self.actionType = actionType 
		self.stochasticType = stochasticType
		self.wind = [0,0,0,1,1,1,2,2,1,0]

		self.start = [3,0]
		self.goal = [3,7]
		self.discount = 1

	def getNextState(self,state,action):
		[r,c] = state
		r += self.wind[c]
		
		if (action == 0 or action == 4 or action == 7): r += 1
		elif (action == 2 or action == 5 or action == 6): r -= 1

		if (action == 1 or action == 4 or action == 5): c += 1
		elif (action == 3 or action == 6 or action == 7): c-= 1
		
		if self.stochasticType == 1: r += np.random.choice([-1,0,1])

		r = max(min(self.numRows-1,r),0)
		c = max(min(self.numCols-1,c),0)
		return [r,c]

	def getAction(self,state,QValue,epsilon):
		if self.actionType == 0: validActions = [0,1,2,3]
		else: validActions = [0,1,2,3,4,5,6,7]
		
		if (np.random.uniform() < epsilon): return np.random.choice(validActions)
		else:
			allActions = [QValue[state[0]][state[1]][a] for a in validActions]
			bestAction = allActions.index(max(allActions))
			return bestAction

	def getEpisodeOutcome(self,QValue,epsilon,alpha):
		iterations = 0
		state = self.start
		action = self.getAction(state,QValue,epsilon)
		while(state != self.goal):
			reward = -1
			nextState = self.getNextState(state,action)
			nextAction = self.getAction(nextState,QValue,epsilon)
			QValue[state[0]][state[1]][action] += alpha*(reward + self.discount*QValue[nextState[0]][nextState[1]][nextAction] - QValue[state[0]][state[1]][action])
			iterations += 1
			state = nextState
			action = nextAction
		return iterations

	def getOptimalPolicy(self,QValue):
		for r in range(self.numRows):
			s = ""
			for c in range(self.numCols):
				s += str(int(QValue[r][c][self.getAction([r,c],QValue,0)])) + " "
			print(s)


	def Sarsa(self,numEpisodes,alpha,file):
		runs = []
		avgSteps = [0]*numEpisodes
		seeds = range(10)
		for seed in seeds:
			np.random.seed(seed)
			QValue = np.zeros((self.numRows,self.numCols,self.numActions))
			totalTime = [0]

			for episode in range(numEpisodes):
				epsilon = 0.5/(episode+1)
				t = self.getEpisodeOutcome(QValue,epsilon,alpha)
				totalTime.append(totalTime[-1] + t)
				avgSteps[episode] += t
			runs.append(totalTime)

			print("Random seed: " + str(seed))
			print("Total steps: ",self.getEpisodeOutcome(QValue,0,alpha))
			self.getOptimalPolicy(QValue)

		runs = np.array(runs)
		avg = np.mean(runs,axis=0)
		avgSteps = [i/10 for i in avgSteps]

		with open(file+'.pkl','wb') as f: pickle.dump(avg,f)
		
		plt.plot(avg,range(avg.shape[0]))
		plt.xlabel('Time steps')
		plt.ylabel('Episodes')
		plt.savefig(file+'.png')

		plt.clf()
		plt.plot(range(len(avgSteps)),avgSteps)
		plt.ylabel('Average Steps')
		plt.xlabel('Episodes')
		plt.savefig(file+'-avgSteps.png')


if __name__ == '__main__':
	actionType = int(sys.argv[1])
	stochasticType = int(sys.argv[2])
	numEpisodes = int(sys.argv[3])
	outputFile = sys.argv[4]
	alpha = 0.5

	gridworld = Gridword(actionType,stochasticType)
	gridworld.Sarsa(numEpisodes,alpha,outputFile)


