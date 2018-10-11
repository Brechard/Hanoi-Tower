class State():

	pins = [1, 2, 3]
	moves = ["s1", "s2", "s3", "b1", "b2", "b3"]
	# If the disks are in different pins, we name the state first with where the big one is
	states = ["b1s1", "b1s2", "b1s3", "s2b2", "s3b3", "b3s2", "b2s3", "b3s3", "b2s2", "b3s1", "b2s1", "s1b1"]

	# If the disks are in different pins, the first disk is the big
	def __init__(self, name, obeyProb):
		self.name = name
		if name[1] == name[3]:	# The two disks are in the same pin 
			if name[0] == "b":		# The botton disk is the big one
				self.onTop = "s"
			else: self.onTop = "b"	
		else: self.onTop = None
		self.b = name[1]
		self.s = name[3]
		self.obeyProb = obeyProb

	def getReward(self):
		reward = -1
		if self.onTop == "b":
			reward = -10
		elif self.s == self.b and int(self.s) == 3:
			reward = 100
		return reward

	# Action 1 to 3 is moving the small to the pin of the number
	# Action 3 to 6 is moving the big to the pin of the number - 3
	def check_move(self, action):

		newState = ""
		if action[0] == "s": # If we are moving the small disk
			if action[1] == self.b: # And the pin where it is going the big one is already there
				onTopAux = "s"
			else: 
				onTopAux  = None
			sAux = action[1]
			bAux = self.b
			newState = "b" +bAux + "s" + sAux 
		else: # We are moving the big disk
			if action[1] == self.s: # if the pin where it is going the small disk is already there
				onTopAux = "b"
				newState = "s" +self.s+ "b" + action[1] 
			else: 
				onTopAux  = None
				newState = "b" +action[1] + "s" + self.s 
			bAux = action[1]
			sAux = self.s
		# The state b3s3 is an end state and the only possibility is to stay there with a reward of 0
		if self.name == newState:
			return 0, newState 
		return State(newState, self.obeyProb).getReward(), newState

	# returns a list of (probability, reward, s') transition tuples
	def get_transition_probs(self, action):

		possibleMoves = []
		reward, newState = self.check_move(action)
		possibleMoves.append((self.obeyProb, reward, newState))
		if self.obeyProb == 1:
			return possibleMoves

		# The state b3s3 is an end state and the only possibility is to stay there with a reward of 0
		if self.name == "b3s3":
			return possibleMoves


		# Get the error move
		if action[0] == "s":
			errorMove = "s" + str([pin for pin in self.pins if pin != int(self.s) and pin != int(action[1])][0])
		else:
			errorMove = "b" + str([pin for pin in self.pins if pin != int(self.b) and pin != int(action[1])][0])
		reward, newState = self.check_move(errorMove)
		possibleMoves.append((1 - self.obeyProb, reward, newState))
		return possibleMoves
