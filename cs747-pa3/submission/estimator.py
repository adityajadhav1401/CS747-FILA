import os, sys
import numpy as np

np.random.seed(1)

class Estimator:
    def __init__(self,path):
        f = open(path)
        data = f.readlines()
        self.numStates = int(data[0].split()[0])
        self.numActions = int(data[1].split()[0])
        self.discountFactor = float(data[2].split()[0])

        self.values = [np.random.rand() for i in range(self.numStates)]
        self.etrace = [0]*self.numStates
        self.trajectory =  [[float(val) for val in line.strip().split('\t')] for line in data[3:]]

    def evaluate(self,path):
        f = open(path)
        data = f.readlines()
        error = 0
        for i in range(self.numStates):
            error += (float(data[i].split()[0]) - self.values[i])**2
        return error

    def estimateTD(self,lam):
        for i in range(len(self.trajectory)-1):
            alphaT = 9.0/(i+1)
            [s, a, r] = [int(self.trajectory[i][0]),int(self.trajectory[i][1]),self.trajectory[i][2]]
            sPrime = int(self.trajectory[i+1][0])

            delta = r + self.discountFactor*self.values[sPrime] - self.values[s]
            self.etrace[s] = self.etrace[s] + 1

            for state in range(self.numStates):
                self.values[state] += alphaT * delta * self.etrace[state]
                self.etrace[state] *= self.discountFactor * lam 

    def estimateMOD(self):
        transitionsEst = {s: {a: {sPrime: 0 for sPrime in range(0,self.numStates)} for a in range(0,self.numActions)} for s in range(0,self.numStates)}
        rewardsEst = {s: {a: {sPrime: 0 for sPrime in range(0,self.numStates)} for a in range(0,self.numActions)} for s in range(0,self.numStates)}

        transitionsTot = {s: {a: {sPrime: 0 for sPrime in range(0,self.numStates)} for a in range(0,self.numActions)} for s in range(0,self.numStates)}
        rewardsTot = {s: {a: {sPrime: 0 for sPrime in range(0,self.numStates)} for a in range(0,self.numActions)} for s in range(0,self.numStates)}
        visitsTot = {s: {a:  0 for a in range(0,self.numActions)} for s in range(0,self.numStates)}
        probsTot = {s: {a: 0 for a in range(0, self.numActions)} for s in range(0, self.numStates)}

        for i in range(len(self.trajectory)-1):
            [s, a, r] = [int(self.trajectory[i][0]),int(self.trajectory[i][1]),self.trajectory[i][2]]
            sPrime = int(self.trajectory[i+1][0])
                
            rewardsTot[s][a][sPrime] += r
            visitsTot[s][a] += 1
            transitionsTot[s][a][sPrime] += 1

        for s in range(0,self.numStates):
            visitsAllActionsTot = sum(visitsTot[s].values())
            for a in range(0, self.numActions):
                for sPrime in range(0,self.numStates):
                    if (visitsTot[s][a] > 0): transitionsEst[s][a][sPrime]=transitionsTot[s][a][sPrime]*1.0/visitsTot[s][a]
                    if (transitionsTot[s][a][sPrime] > 0): rewardsEst[s][a][sPrime]=rewardsTot[s][a][sPrime]*1.0/transitionsTot[s][a][sPrime]

                probsTot[s][a] = visitsTot[s][a]*1.0/visitsAllActionsTot

        self.values = [0]*self.numStates
        while True:
            valuesNew = [0]*self.numStates
            for s in range(0,self.numStates):
                for a in range(0,self.numActions):    
                    for sPrime in range(0,self.numStates):
                        valuesNew[s] += probsTot[s][a]*transitionsEst[s][a][sPrime]*(rewardsEst[s][a][sPrime]+self.discountFactor*self.values[sPrime])

            if (np.max(np.abs(np.array(valuesNew)-np.array(self.values))) < 1e-16): break   
            self.values = valuesNew


    def print(self):
        for i in range(self.numStates):
            print(self.values[i])


if __name__ == '__main__':
    tracePath = ""
    solutionPath = ""
    for i in range(len(sys.argv)):
        if (sys.argv[i] == "--trace"):
            tracePath = sys.argv[i+1]
        elif (sys.argv[i] == "--solution"):
            solutionPath = sys.argv[i+1]
    
    if tracePath == "":
        print("Input tracefile not given")
        exit(0)
    if not os.path.isfile(tracePath): 
        print(tracePath + " : No such file exists")
        exit(0)
    if (not solutionPath == "" and not os.path.isfile(solutionPath)): 
        print(solutionPath + " : No such file exists")
        exit(0)

    

    #=========#
    # General #
    #=========#

    # TD Lambda
    # lambdas = [float(i)/100 for i in range(50,100,1)]
    # errors = []
    # for lam in lambdas:
    #     estimator = Estimator(tracePath)
    #     estimator.estimateTD(lam)
    #     errors.append(estimator.evaluate(solutionPath))
    # i = errors.index(min(errors))
    # estimator = Estimator(tracePath)
    # estimator.estimateTD(lambdas[i])
    # estimator.print()

    # MODEL BASED 
    estimator = Estimator(tracePath)
    estimator.estimateMOD()
    estimator.print()



    #=========#
    # Testing #
    #=========#

    # TD Lambda
    # lambdas = [float(i)/100 for i in range(50,100,1)]
    # errors = []
    # for lam in lambdas:
    #     estimator = Estimator(tracePath)
    #     estimator.estimateTD(lam)
    #     errors.append(estimator.evaluate(solutionPath))
    # print(min(errors))
    
    
    # MODEL BASED
    # estimator = Estimator(tracePath)
    # estimator.estimateMOD()
    # print(estimator.evaluate(solutionPath))




