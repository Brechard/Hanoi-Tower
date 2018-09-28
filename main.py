from state import State
import random
# If the disks are in different pins, we name the state first with where the big one is
statesString = ["b1s1", "b1s2", "b1s3", "s2b2", "s3b3", "b3s2", "b2s3", "b3s3", "b2s2", "b3s1", "b2s1", "s1b1"]
obeyProb = 0.9
moves = ["s1","s2","s3", "b1", "b2","b3"]
V = {}
pi = {}
GAMMA = 0.9
SMALL_ERROR = 1e-3

# Creation of every state object
states = []
for state in statesString:
	states.append(State(state, obeyProb))

state_to_action = {
    states[0].name: ["s2", "s3"],
    states[1].name: ["s1", "s3", "b2", "b3"],
    states[2].name: ["s1", "s2", "b2", "b3"],
    states[3].name: ["b1", "b3"],
    states[4].name: ["b1", "b2"],
    states[5].name: ["s1", "s3", "b1", "b2"],
    states[6].name: ["s1", "s2", "b1", "b3"],
    states[7].name: ["s3"],
    states[8].name: ["s1", "s3"],
    states[9].name: ["s2", "s3", "b1", "b2"],
    states[10].name: ["s2", "s3", "b1", "b3"],
    states[11].name: ["b2", "b3"]
}

def calculate_best_action_values(state):
	best_value = float('-inf')
	best_action = None
	# print()
	# print("State:  " +state.name)
	for action in state_to_action.get(state.name, []):
		expected_value = 0
		# print("For the action " +action)
		for probability, reward, state_prime in state.get_transition_probs(action):
			# print("state_prime " +state_prime+ " with prob " +str(probability)+ " and reward " +str(reward))
			# print("the new value is: " +str(probability * (reward + GAMMA * V[state_prime])))
			expected_value += probability * (reward + GAMMA * V[state_prime])
			# print("and the new expected_value is: " +str(expected_value))

		if expected_value > best_value:
			best_value = expected_value
			best_action = action

	# print("The best value to return is: " +str(best_value))
	return best_action, best_value

def value_iteration():
	for state in states:
		V[state.name] = 0
		pi[state.name] = ""
	delta = 1
	while delta > SMALL_ERROR:
		delta = 0
		for state in states:
			new_best_action, new_best_value = calculate_best_action_values(state)
			if abs(new_best_value - V[state.name]) > delta:
				delta = abs(new_best_value - V[state.name])

			V[state.name] = new_best_value
			pi[state.name] = new_best_action
	for key, value in V.items():
		print("State " +key+ " policy is: " +pi[key]+ " with a value: " +str(value))
	# print(V)

def initialize_random_policy():
  	# policy is a lookup table for state -> action
	print("Random policy: ")
	policy = {}

	for state in states:
		possible_actions = state_to_action.get(state.name, [])
		policy[state.name] = possible_actions[random.randint(0, len(possible_actions) - 1)]
		V[state.name] = 0
	print(policy)
	return policy

def policy_iteration():
  	# First we create a random policy that we will update as we learn
	policy = initialize_random_policy()
	# Solve V[s] = ∑s'∈S P(s'|s,π[s])(R(s,a,s')+γV[s'])
	# V[s] = sum([P[s,policy[s],s1] * (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])
	changed = True
	while changed:
		changed = False
		
		# Calculate the utility considering the actual policy
		for state in states:
			expected_value = 0

			for probability, reward, state_prime in state.get_transition_probs(policy[state.name]):
				expected_value += probability * (reward + GAMMA * V[state_prime])

			V[state.name] = expected_value 	

		# Check if the best action is the same as the actual policy
		for state in states:
			best_action, _ = calculate_best_action_values(state)
			
			if policy[state.name] != best_action:
				policy[state.name] = best_action
				changed = True

	print("Optimal policy: ")
	for key, value in policy.items():
		print("State " +key+ " policy: " +str(value))

value_iteration()
print()
policy_iteration()