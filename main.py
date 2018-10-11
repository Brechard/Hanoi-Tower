from state import State
import random
import timeit
import matplotlib.pyplot as plt
import statistics as stats

# If the disks are in different pins, we name the state first with where the big one is
statesString = ["b1s1", "b1s2", "b1s3", "s2b2", "s3b3", "b3s2", "b2s3", "b3s3", "b2s2", "b3s1", "b2s1", "s1b1"]
obeyProb = 0.9
moves = ["s1","s2","s3", "b1", "b2","b3"]
V = {}
V_prime = {}
pi = {}
GAMMA = 0.9
policy = {}
value = {}
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
	# if(state.name == "b1s3"):
	# print()
	# print()
	# print()
	# print("calculate_best_action_values: State " +state.name)
	for action in state_to_action.get(state.name, []):
		# if(state.name == "b1s3"):
		# print()
		# print()
		# print("considering now the action: " +action)
		expected_value = 0
		for probability, reward, state_prime in state.get_transition_probs(action):
			# if(state.name == "b1s3"):
			# print("state_prime " +state_prime+ " with prob " +str(round(probability, 5))+ " and reward " +str(reward))
			# print("And value: " +str(round(V[state_prime], 3)))
			# print("the new value is: " +str(round(probability * (reward + GAMMA * V[state_prime]), 3)))
			expected_value += probability * (reward + GAMMA * V[state_prime])
			# if(state.name == "b1s3"):
			# print("the new expected_value is: " +str(round(expected_value, 3)))
			# print()
		# if(state.name == "b1s3"):
		# print("expected_value: " +str(expected_value)+ " best_value: " +str(best_value))

		if expected_value > best_value:
			best_value = expected_value
			best_action = action

	# print("The best value to return is: " +str(best_value))
	return best_action, best_value

def value_iteration(error):
	deltas = []
	for state in states:
		V[state.name] = 0
		V_prime[state.name] = 0
		pi[state.name] = ""
	delta = 1
	while delta > error:
		delta = 0
		delta_prime = 0
		for state in states:

			new_best_action, new_best_value = calculate_best_action_values(state)

			if abs(new_best_value - V[state.name]) > delta:
				delta = abs(new_best_value - V[state.name])
			V_prime[state.name] = new_best_value
			pi[state.name] = new_best_action
	
			# deltas.append(delta)

		for key, value in V_prime.items():
			V[key] = value


	# plt.plot(deltas)
	# plt.show()
	for key, value in V.items():
		print("State " +key+ " policy: " +pi[key]+ "  value: " +str(round(value, 2)))

def initialize_random_policy():
  	# policy is a lookup table for state -> action
	# print("Random policy: ")

	for state in states:
		possible_actions = state_to_action.get(state.name, [])
		policy[state.name] = possible_actions[random.randint(0, len(possible_actions) - 1)]
		V[state.name] = 0
		V_prime[state.name] = 0
		value[state.name] = 0
	# print(policy)
	return policy

def policy_iteration():
  	# First we create a random policy that we will update as we learn
	policy = initialize_random_policy()
	# Solve V[s] = ∑s'∈S P(s'|s,π[s])(R(s,a,s')+γV[s'])
	# V[s] = sum([P[s,policy[s],s1] * (R[s,policy[s],s1] + gamma*V[s1]) for s1 in range(N_STATES)])
	changed = True
	# print("V" +str(V))
	while changed:
		changed = False
		print()
		for state in states:
			expected_value = 0

			for probability, reward, state_prime in state.get_transition_probs(policy[state.name]):
				
				expected_value += probability * (reward + GAMMA * V[state_prime])

			V[state.name] = expected_value 	
			print("For the state " +state.name+ " the expected_value is: " +str(expected_value))

		# Check if the best action is the same as the actual policy
		for state in states:

			best_action, best_value = calculate_best_action_values(state)
			print("The best action is " +best_action+ " and best_value is: " +str(best_value))
			
			if policy[state.name] != best_action:
				policy[state.name] = best_action
				value[state.name] = best_value
				changed = True
			else: value[state.name] = best_value

	print("Optimal policy: ")
	for key, v in policy.items():
		print("State " +key+ " policy: " +str(v)+ " value: " +str(round(value[key], 2)))

def check_error_policy_value(times):
	V_Aux = {}
	for x in range(1, times):
		value_iteration(0.001)
		policy_iteration()
		# equals = True
		for key, value in policy.items():
			# equals = equals and pi[key] == value
			if not(pi[key] == value):
				break
				try: V_Aux[key] = V_Aux[key] + 1
				except KeyError: V_Aux[key] = 1  
				# if key == "b1s3":
				# 	return
					# print("State " +key+ " -> value_iteration: " +pi[key]+ " policy_iteration: " +value)
		# print(equals)
	print(V_Aux)
	for key, value in V_Aux.items():
		print("For state " +key+ " there have been " +str(value)+ " errors")

# check_error_policy_value(10000)
value_iteration(0.001)
print()
policy_iteration()

def calculate_convergence_speed(errors, n_times):
    times = []
    for error in errors:
        times.append(calculate_avg_time_value_iteration(error, n_times))
        # print(str(times[len(times) - 1])+ "s for error: " +str(error))

    plt.plot(errors, times)
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.legend(errors)

    plt.show()
    
def calculate_avg_time_value_iteration(error, times):
    time = 0
    t = []
    for x in range(1, times):
        start = timeit.default_timer()
        value_iteration(error)
        final = timeit.default_timer()
        t.append(final - start)

    print(str(round(stats.median(t),6)) + "s for error " +str(round(error,4)))
    return stats.median(t)

# list_errors = []
# for x in range(1, 100):
#     list_errors.append(1 - x/100)
# list_errors.append(frange(0.5, 1, 0.01))
# list_errors.append([0.5:0.01: 1])
# print(str(list_errors))
# list_errors = [2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.01, 0.001]
# calculate_convergence_speed(list_errors, 100)
# value_iteration(0.001)
