### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    #Initializing Value Function
    value_function = np.zeros(nS)

    j = 0
    #Loop for convergence 
    while (True):

      #Initialising delta = 0  - Initizlized before calculating for all of states      
      delta = 0

      #Loop inside all states
      for state in range(nS):
        actions_for_state = policy[state]  #Action vector for a particular state - Prob of taking each action (1x4)
        old_value_state = value_function[state] #Storing Old state value
        new_value_state = 0 #Initializing New State Value = 0, for every state (for every convergence loop) - Initizlized before calculating for each action - Thus calculated for eachs state

        #Loop inside no. of actions
        for action in range(nA) :
          prob = actions_for_state[action]  #Prob of that particular action (for that particular state)
          directions = len(P[state][action]) #Since stochastic (each action leads to taking different states (in diff directions - 3 or 1))

          #Loop into each direction, i.e all of next state
          for i in range(directions):
            trans_prob , next_state , reward , end = P[state][action][i] #Storing transition_probability, next state and reward -for that particular direction)
            value = prob*trans_prob*(reward + gamma*value_function[next_state]) #State value calulated (In loop for each state, action and sub direction)
            new_value_state += value #New state value found, for each action (also summed over all sub-directions) and summed over all actions associated for that particular state

        #Once for each state, the state value is calculated (Summer over all actions), the absolute diff is calucalted between new and prev state value
        #This is looped under all of states  
        diff = abs(new_value_state-old_value_state)
        #print("diff = ", diff)

        #Delta is calculated for each state, and updated on the maximum between delta in all of state - that is only for one iteration, then it is reinitialsed to 0
        delta = max(delta, diff)  
        #print("delta = ", delta)

        #New Value function updated for each state, in each iteration
        value_function[state] = new_value_state
      
      #After one entire iteration for all of states, delta for that iteration caluculated for all of state is checked with threshold
      if(delta<tol):
        break    #If lesser than threshold, then break the loop        
      else :
        j = j+1  #Counting no. of iterations
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################
    return new_policy

def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    old_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution

    new_policy = np.zeros([nS, nA]) #Initialize new_policy matrix
    new_value = np.zeros([nS, nA])  #Initialize new_policy value matrix
    #Looping for each state
    for state in range(nS):

      #Looping for each action
      for action in range(nA) :   
        new_value_action = 0      #Setting new value action as 0 - Initilizing before going into sub-direction for each action - Thus calculated for each action
        directions = len(P[state][action])  #Calulating no. of sub-direction for each action

        #Looping thriugh each sub-direction
        for i in range(directions):
          trans_prob , next_state , reward , end = P[state][action][i] #Storing transition_probability, next state and reward -for that particular direction)
          value = trans_prob*(reward + gamma*value_from_policy[next_state]) # State value calulated (In loop for each state, action and sub direction)
          new_value_action += value       #New state value found, for each action (also summed over all sub-directions)
        new_value[state][action] = new_value_action #Value stored for each action and each state (nS x nS)
      

    #After loopoing through the entire state and filling the new_value matrix for (nS, nA)  
    index = np.argmax(new_value , axis = 1) #Calualting index for each state(row) which has max value (corres to action corresp to that state)
    rows = np.arange(new_value.shape[0])
    new_policy[rows, index] = 1 # Creating a one-hot vector type, 1- corr to action to be taken for that state

    return new_policy

def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """

    #Setting given poilicy as old policy, to be evalited 
    old_policy = policy.copy()

    #Setting flag as 1
    k = 1    

    #Looping untl falg remains 1
    while (k==1):
      value_from_policy = policy_evaluation(P, nS, nA, old_policy, gamma=0.9, tol=1e-8) #Evalute the state values for a given old_policy
      new_policy = policy_improvement(P, nS, nA, value_from_policy, gamma=0.9)  #Look for improvement in policy given the state value wrt the old policy

      #If the old policy and new policy are same, then set k flag 0 and break out of loop
      if np.array_equal(new_policy, old_policy):   
        k = 0 

      #If improved policy is not equal to old policy, replace the imporved policy as old policy
      else :
        old_policy = new_policy  
        
    return new_policy, value_from_policy

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    old_value_function = V.copy()
    policy_new = np.zeros([nS, nA])

    #old_value_function = np.zeros(nS)
    new_value = np.zeros([nS, nA])
    new_policy = np.zeros([nS, nA])

    while (True):
      for state in range (nS):

        for action in range (nA):
          new_state_action_value = 0
          directions = len(P[state][action])

          for i in range (directions) :

            trans_prob , next_state, reward, end = P[state][action][i]
            value = trans_prob*(reward + gamma*old_value_function[next_state])
            new_state_action_value +=value

          new_value[state][action] = new_state_action_value

      new_value_function = np.max(new_value, axis=1)
      delta = np.max(np.abs(new_value_function - old_value_function))

      if (delta<tol):
        break
      else:
        old_value_function = new_value_function  

    index = np.argmax(new_value , axis = 1)
    rows = np.arange(new_value.shape[0])
    new_policy[rows, index] = 1

    return new_policy, new_value_function



def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        observation , _ = env.reset() # initialize the episode
        done = False
        next_state = observation
        while not done: # using "not truncated" as well, when using time_limited wrapper. (Loop wil keep on running until done = 'True')
            if render:
                env.render() # render the game

            #Get the action from the current state, based on the policy
            action = policy[next_state]  
            #Get the action index 
            action_index = np.where(action ==1)[0][0]
            #Get the information info of next_state, reward, done and prob from the environement given the action
            next_state, reward, done, info, prob = env.step(action_index)

            #Calculate total reward at each step and episode
            total_rewards += reward
                    
    return total_rewards


