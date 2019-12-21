# Markov Decision Process 
# Grid world example
# code adopted from UMich EECS 492, Fall 2019
# Qianbo Yin

# usage: python3 grid_mdp.py [file_name.txt]



import sys
import copy


class node:
	'''
	This class is for book-keeping each state in the MDP.

	self.util is the utility of this state
	self.reward is R(s), the reward of getting to this state
	self.isWall is a flag that indicates whether this cell is occupied by wall
	self.isTerminal is a flag that indicates whether this state is a terminal state.

	'''
	def __init__(self, util = 0, reward=-0.04, isWall=False, isTerminal=False):
		self.util = util
		self.reward = reward
		self.isWall = isWall
		self.isTerminal = isTerminal

class grid:
	'''
	This class represents our MDP problem. It consists of

	- Description of the environment
	- Method to print the grid
	- Method to do Value Iteration (you need to implement)

	'''

	def __init__(self, gridfile):
		'''
		This init function helps you read and parse the environment file.
		And it maintains the below data members:

		self.gamma : discount factor
		self.living_cost : "cost of living", a negative reward
		self.mprobs : transition probabilities
		self.grid : a rectangle (defined by nrows-by-ncols) that represents 
					the environment, with i,j-th element being a node representing the thing at i,j
		'''
		line = open(gridfile, 'r').readlines()
		self.gamma = float(line[0])
		self.living_cost = float(line[1])
		mprobs = line[2].split()
		self.mprobs = [float(n) for n in mprobs]
		self.grid = []
		for row in line[3:]:
			gridrow = []
			splitrow = row.split()
			if len(splitrow) == 0:
				continue
			for ch in row.split():
				if ch == '*':
					gridrow.append(node(reward=self.living_cost))
				elif ch == 'x':
					gridrow.append(node(isWall=True))
				else:
					try:
						utility = float(ch)
						gridrow.append(node(util=utility, isTerminal=True))	
					except ValueError:
						print ("Bad grid value")
						sys.exit()
			self.grid.append(gridrow)
		self.nrows = len(self.grid)
		assert(self.nrows > 0)
		self.ncols = len(self.grid[0])
		assert(self.ncols > 0)

	def printGrid(self):
		'''
		Print the grid
		'''
		printstr = ''
		for row in self.grid:
			for c in row:
				if not c.isWall:
					printstr += str(round(c.util, 3)) + ' '
				else:
					printstr += 'x '
			printstr += '\n'
		print (printstr)

	def is_coord_open(self, i, j):
		'''
		Checks if there is a grid space at (i,j), and whether or not there is a wall there 
		Returns True if there is a free grid space
		Returns False otherwise
		''' 
		if i >= 0 and j < self.ncols and i < self.nrows and j >= 0 and not self.grid[i][j].isWall:
			return True
		else:
			return False


	def doValueIteration(self, epsilon=0.00001):
		'''
		This function modifies the utilities of each cell in the grid.
		
		Input:
			epsilon : the maximum error allowed in the utility of any state
			self: this MDP problem
		
		Output:
			No need to return anything. You just need to modify the utility for 
			each cell (state) in self.grid.
		'''
		#DONE: Complete value iteration!
		delta = 1
		convergence = min((1-self.gamma)/self.gamma, 1)
		if convergence == 0:
			convergence = 1
		niters = 0
		while delta >= epsilon * convergence:
			delta = 0
			new_grid = copy.deepcopy(self.grid)
			for i in range(self.nrows):
				for j in range(self.ncols):
					cur = self.grid[i][j]
					if cur.isWall or cur.isTerminal:
						continue
					# find resulting position in each direction
					# up state
					if self.is_coord_open(i-1, j):
						up_state = self.grid[i-1][j]
					else:
						up_state = cur
					# right state
					if self.is_coord_open(i, j+1):
						right_state = self.grid[i][j+1]
					else:
						right_state = cur
					# down state
					if self.is_coord_open(i+1, j):
						down_state = self.grid[i+1][j]
					else:
						down_state = cur
					# left state
					if self.is_coord_open(i, j-1):
						left_state = self.grid[i][j-1]
					else:
						left_state = cur
					
					# find Q-value of each action
					north_qval = self.mprobs[0]*up_state.util + self.mprobs[1]*right_state.util + self.mprobs[2]*down_state.util + self.mprobs[3]*left_state.util
					south_qval = self.mprobs[0]*down_state.util + self.mprobs[1]*left_state.util + self.mprobs[2]*up_state.util + self.mprobs[3]*right_state.util
					east_qval =  self.mprobs[0]*right_state.util + self.mprobs[1]*down_state.util + self.mprobs[2]*left_state.util + self.mprobs[3]*up_state.util
					west_qval =  self.mprobs[0]*left_state.util + self.mprobs[1]*up_state.util + self.mprobs[2]*right_state.util + self.mprobs[3]*down_state.util
					
					# bellman update for utility
					new_grid[i][j].util = cur.reward + self.gamma * max(north_qval, south_qval, east_qval, west_qval)
					# keep track of maximum change, delta
					change = abs(new_grid[i][j].util - cur.util)
					if change > delta:
						delta = change

			self.grid = new_grid
			niters += 1

		
			
if __name__=='__main__':
	assert(len(sys.argv)>1)
	g = grid(sys.argv[1])
	g.doValueIteration()
	g.printGrid()

