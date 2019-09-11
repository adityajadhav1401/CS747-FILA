import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = open('outputCopy.txt','r')
lines = file.readlines()

instances = {"../instances/i-1.txt":0,"../instances/i-2.txt":1,"../instances/i-3.txt":2}
algorithms = {"round-robin":0,"epsilon-greedy":1,"ucb":4,"kl-ucb":5,"thompson-sampling":6}
epsilons = {"-":0,"0.002":0,"0.02":1,"0.2":2}
horizons = {"50":0,"200":1,"800":2,"3200":3,"12800":4,"51200":5,"204800":6}

data = np.zeros([len(instances),len(algorithms)+2,len(horizons)])

for line in lines:
	vals = line.split(',')
	data[instances[vals[0]],algorithms[vals[1]]+epsilons[vals[3]],horizons[vals[4]]] += float(vals[5]) / 50.0


# print(data)
# exit(0)


# instance 1
x = np.asarray([50,200,800,3200,12800,51200,204800])
y1 = data[0,0,:]
y2 = data[0,1,:]
y3 = data[0,2,:]
y4 = data[0,3,:]
y5 = data[0,4,:]
y6 = data[0,5,:]
y7 = data[0,6,:]
plt.title('instance 1')
plt.xscale('log')
#plt.plot(x,y1, marker='o', markerfacecolor='#E74C3C', markersize=5, color='#EC7063', linewidth=2, label="round-robin",alpha=0.7)
# plt.plot(x,y2, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2, label="epsilon-greedy (0.002)",alpha=0.7)
# plt.plot(x,y3, marker='o', markerfacecolor='#3498DB', markersize=5, color='#5DADE2', linewidth=2, label="epsilon-greedy (0.02)",alpha=0.7)
# plt.plot(x,y4, marker='o', markerfacecolor='#27AE60', markersize=5, color='#52BE80', linewidth=2, label="epsilon-greedy (0.2)",alpha=0.7)
plt.plot(x,y5, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2, label="ucb",alpha=0.7)
plt.plot(x,y6, marker='o', markerfacecolor='#E67E22', markersize=5, color='#EB984E', linewidth=2, label="kl-ucb",alpha=0.7)
plt.plot(x,y7, marker='o', markerfacecolor='#34495E', markersize=5, color='#5D6D7E', linewidth=2, label="thompson-sampling",alpha=0.7)
plt.legend()
# plt.show()
plt.savefig('instance1.png',bbox_inches='tight')
plt.clf()

# instance 2
x = np.asarray([50,200,800,3200,12800,51200,204800])
y1 = data[1,0,:]
y2 = data[1,1,:]
y3 = data[1,2,:]
y4 = data[1,3,:]
y5 = data[1,4,:]
y6 = data[1,5,:]
y7 = data[1,6,:]
plt.title('instance 2')
plt.xscale('log')
#plt.plot(x,y1, marker='o', markerfacecolor='#E74C3C', markersize=5, color='#EC7063', linewidth=2, label="round-robin",alpha=0.7)
# plt.plot(x,y2, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2, label="epsilon-greedy (0.002)",alpha=0.7)
# plt.plot(x,y3, marker='o', markerfacecolor='#3498DB', markersize=5, color='#5DADE2', linewidth=2, label="epsilon-greedy (0.02)",alpha=0.7)
# plt.plot(x,y4, marker='o', markerfacecolor='#27AE60', markersize=5, color='#52BE80', linewidth=2, label="epsilon-greedy (0.2)",alpha=0.7)
plt.plot(x,y5, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2, label="ucb",alpha=0.7)
plt.plot(x,y6, marker='o', markerfacecolor='#E67E22', markersize=5, color='#EB984E', linewidth=2, label="kl-ucb",alpha=0.7)
plt.plot(x,y7, marker='o', markerfacecolor='#34495E', markersize=5, color='#5D6D7E', linewidth=2, label="thompson-sampling",alpha=0.7)
plt.legend()
# plt.show()
plt.savefig('instance2.png',bbox_inches='tight')
plt.clf()

# instance 3
x = np.asarray([50,200,800,3200,12800,51200,204800])
y1 = data[2,0,:]
y2 = data[2,1,:]
y3 = data[2,2,:]
y4 = data[2,3,:]
y5 = data[2,4,:]
y6 = data[2,5,:]
y7 = data[2,6,:]
plt.title('instance 3')
plt.xscale('log')
#plt.plot(x,y1, marker='o', markerfacecolor='#E74C3C', markersize=5, color='#EC7063', linewidth=2, label="round-robin",alpha=0.7)
# plt.plot(x,y2, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2, label="epsilon-greedy (0.002)",alpha=0.7)
# plt.plot(x,y3, marker='o', markerfacecolor='#3498DB', markersize=5, color='#5DADE2', linewidth=2, label="epsilon-greedy (0.02)",alpha=0.7)
# plt.plot(x,y4, marker='o', markerfacecolor='#27AE60', markersize=5, color='#52BE80', linewidth=2, label="epsilon-greedy (0.2)",alpha=0.7)
plt.plot(x,y5, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2, label="ucb",alpha=0.7)
plt.plot(x,y6, marker='o', markerfacecolor='#E67E22', markersize=5, color='#EB984E', linewidth=2, label="kl-ucb",alpha=0.7)
plt.plot(x,y7, marker='o', markerfacecolor='#34495E', markersize=5, color='#5D6D7E', linewidth=2, label="thompson-sampling",alpha=0.7)
plt.legend()
# plt.show()
plt.savefig('instance3.png',bbox_inches='tight')
plt.clf()