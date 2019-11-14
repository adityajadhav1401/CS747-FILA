Use the following command to create all the plots - 
./script.sh


Files in the directory -
gridworld.py : 	Used to create plot for the given onstance of gridworld. 
				To use the file, enter the following command.
				python3 gridworld.py <isKings> <isStochastic> <numEpisodes>

				<isKing> will be 0 or 1 if you want to use, not want to use king's moves respectively.
				<isStochastic>  will be 0 or 1 if you want to use, not want to use stochastic rain respectively.

Report.pdf 				:	Explains all the observations
Task(i).pkl				:	Console  output when we run gridworld.py
Task(i).png				:	Plot of "Episodes" v/s "Timesteps"
Task(i)-avgSteps.png	:	Plot of "Avg Timesteps" v/s "Episodes"