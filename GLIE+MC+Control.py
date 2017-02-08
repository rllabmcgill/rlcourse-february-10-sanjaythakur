
# coding: utf-8

# * GLIE
# * Model Free, used only state action values
# * Every Visit MC
# * Easily extensible to any TD method by changing value of LAMBDA

# In[1]:

import random
import os
import time


# Defining the constants here

# In[2]:

INFINITY = 1000


# User controlled variables and other hyper-parameters

# In[3]:

#LAMBDA as in controlling the depth of TD. Making it INFINITY tantamounts to MC, and making it 0 tantamounts to TD(0). You can make it anything in between to get TD(lambda) algorithm. Remember, decreasing it would increase bias and reducing variance. 
LAMBDA = INFINITY

#Discount-Factor
GAMMA = 0.9

#All possible actions defined
ACTION_UP = 'UP'
ACTION_DOWN = 'DOWN'
ACTION_LEFT = 'LEFT'
ACTION_RIGHT = 'RIGHT'

#Number of episodes to consider to evaluate each policy
NUMBER_OF_EPISODES_PER_POLICY_EVALUATION = 1

#Maximum number of iterations for convergence to the optimal policy
MAXIMUM_NUMBER_OF_POLICY_ITERATIONS = 40

#Start and end of any episode
START_STATE = '00'
END_STATE = '15'

#Defining the EPSILON which would ensure regular exploration. Our EPSILON will decrease linearly with each iteration of a episode and will eventually fade away to 0 .
EPSILON = 1

#Maximum allowed episode length
MAXIMUM_EPISODE_LENGTH = 100

#WAIT TIME
wait_time = 2

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)


# The following section defines the MDP

# In[4]:

all_states = ['00', '01', '02', '03',
          '04', '05', '06', '07',
          '08', '09', '10', '11',
          '12', '13', '14', '15']

immediate_state_rewards = {'00': -1,'01': -1,'02': -1,'03': -1,
                           '04': -1,'05': -1,'06': -1,'07': -1,
                           '08': -1,'09': -1,'10': -1,'11': -1,
                           '12': -1,'13': -1,'14': -1,'15': 0 
                          }

all_transitions =  {
    '00': {ACTION_UP : '00', ACTION_RIGHT : '01', ACTION_DOWN: '04', ACTION_LEFT: '00'},
    '01': {ACTION_UP : '01', ACTION_RIGHT : '02', ACTION_DOWN: '05', ACTION_LEFT: '00'},
    '02': {ACTION_UP : '02', ACTION_RIGHT : '03', ACTION_DOWN: '06', ACTION_LEFT: '01'},
    '03': {ACTION_UP : '03', ACTION_RIGHT : '03', ACTION_DOWN: '07', ACTION_LEFT: '02'},
    '04': {ACTION_UP : '00', ACTION_RIGHT : '05', ACTION_DOWN: '08', ACTION_LEFT: '04'},
    '05': {ACTION_UP : '01', ACTION_RIGHT : '06', ACTION_DOWN: '09', ACTION_LEFT: '04'},
    '06': {ACTION_UP : '02', ACTION_RIGHT : '07', ACTION_DOWN: '10', ACTION_LEFT: '05'},
    '07': {ACTION_UP : '03', ACTION_RIGHT : '07', ACTION_DOWN: '11', ACTION_LEFT: '06'},
    '08': {ACTION_UP : '04', ACTION_RIGHT : '09', ACTION_DOWN: '12', ACTION_LEFT: '08'},
    '09': {ACTION_UP : '05', ACTION_RIGHT : '10', ACTION_DOWN: '13', ACTION_LEFT: '08'},
    '10': {ACTION_UP : '06', ACTION_RIGHT : '11', ACTION_DOWN: '14', ACTION_LEFT: '09'},
    '11': {ACTION_UP : '07', ACTION_RIGHT : '11', ACTION_DOWN: '15', ACTION_LEFT: '10'},
    '12': {ACTION_UP : '08', ACTION_RIGHT : '13', ACTION_DOWN: '12', ACTION_LEFT: '12'},
    '13': {ACTION_UP : '09', ACTION_RIGHT : '14', ACTION_DOWN: '13', ACTION_LEFT: '12'},
    '14': {ACTION_UP : '10', ACTION_RIGHT : '15', ACTION_DOWN: '14', ACTION_LEFT: '13'},
    '15': {ACTION_UP : '15', ACTION_RIGHT : '15', ACTION_DOWN: '15', ACTION_LEFT: '15'},
}


# We would tweak the following entities during the course of our quest of finding the optimal policy and state action values

# In[5]:

all_state_action_value_pairs = {
    '00': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '01': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '02': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '03': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '04': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '05': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '06': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '07': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '08': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '09': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '10': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '11': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '12': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '13': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '14': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '15': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
}

# We initialize our policy. Our first iteration of policy iteration would anyways be uniformly random as we have initialized EPSILON as 1, and that is the nature of GLIE. After first policy iteration where we would be tweaking our policy greedily we'll end up having a deterministic policy.
policy = {
    '00': ACTION_UP,
    '01': ACTION_UP,
    '02': ACTION_UP,
    '03': ACTION_UP,
    '04': ACTION_UP,
    '05': ACTION_UP,
    '06': ACTION_UP,
    '07': ACTION_UP,
    '08': ACTION_UP,
    '09': ACTION_UP,
    '10': ACTION_UP,
    '11': ACTION_UP,
    '12': ACTION_UP,
    '13': ACTION_UP,
    '14': ACTION_UP,
    '15': ACTION_UP,
}

total_state_action_visits = {
    '00': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '01': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '02': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '03': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '04': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '05': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '06': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '07': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '08': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '09': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '10': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '11': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '12': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '13': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '14': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
    '15': {ACTION_UP : 0, ACTION_RIGHT : 0, ACTION_DOWN: 0, ACTION_LEFT: 0},
}


# General-purpose functions and algorithm specific functions

# In[6]:

def printPolicy():
    print("Updated Policy", end = '')
    for state in all_states:
        if (int(state) % 4) == 0:
            print("\n")
        print(state, "::", policy[state],"\t", end = '')
    print("\n\n")

def printStateActionValuePairs():
    print(" \t", ACTION_UP, "\t", ACTION_RIGHT, "\t", ACTION_DOWN, "\t", ACTION_LEFT)
    for state in all_states:
        print(state, "\t", "%.2f" % all_state_action_value_pairs[state][ACTION_UP], "\t", "%.2f" % all_state_action_value_pairs[state][ACTION_RIGHT], "\t", "%.2f" % all_state_action_value_pairs[state][ACTION_DOWN], "\t", "%.2f" % all_state_action_value_pairs[state][ACTION_LEFT],)
    print("\n\n")

def printGridWorld(states_in_episode = [], actions_in_episode = []):
    for state in all_states:
        if (int(state) % 4) == 0:
            print("\n")
        if state in states_in_episode:
            state = state.replace(state, GREEN(state))
        print(state, "\t", end = '')
        
    print("\n")
    #print('\n', 'All Actions until this time')
    
    #for action in actions_in_episode:
    #    print(action, "\t", end = '')
        
def stateActionBasedDiscountedReturn(states_in_episode, actions_in_episode):
    estimated_value = 0.0
    effective_discounting = 1.0
    for step_iterator in range(len(states_in_episode)):
        if step_iterator > LAMBDA:
            break
        estimated_value = estimated_value + (effective_discounting * (immediate_state_rewards[states_in_episode[step_iterator]]))
        effective_discounting = effective_discounting * GAMMA
    estimated_value = estimated_value + (effective_discounting * (all_state_action_value_pairs[states_in_episode[step_iterator]][actions_in_episode[step_iterator]]))
    return estimated_value


# Functions for random sampling

# In[7]:

def updatePolicy():
    for state, action_values in all_state_action_value_pairs.items():
        highest_valued_action = ACTION_UP
        highest_value = action_values[ACTION_UP]
        if highest_value < action_values[ACTION_RIGHT]:
            highest_valued_action = ACTION_RIGHT
            highest_value = action_values[ACTION_RIGHT]
        if highest_value < action_values[ACTION_DOWN]:
            highest_valued_action = ACTION_DOWN
            highest_value = action_values[ACTION_DOWN]
        if highest_value < action_values[ACTION_LEFT]:
            highest_valued_action = ACTION_LEFT
            highest_value = action_values[ACTION_LEFT]
            
        policy[state] = highest_valued_action

def chooseActionForRandomSampling():
    random_throw = random.uniform(0, 1)
    if random_throw < 0.25:
        return ACTION_UP
    elif random_throw < 0.5:
        return ACTION_RIGHT
    elif random_throw < 0.75:
        return ACTION_DOWN
    else:
        return ACTION_LEFT

def generateRandomlySampledEpisode():
    current_state = START_STATE
    states_in_episode = []
    actions_in_episode = []
    while (current_state != END_STATE) & (len(states_in_episode) < MAXIMUM_EPISODE_LENGTH):
        states_in_episode.append(current_state)
        action = chooseActionForRandomSampling()
        actions_in_episode.append(action)
        
        #os.system('clear')
        #printGridWorld(states_in_episode, actions_in_episode)
        #time.sleep(wait_time)
        
        current_state = all_transitions.get(current_state).get(action)
        
    if current_state == END_STATE:
        states_in_episode.append(END_STATE)
        actions_in_episode.append(chooseActionForRandomSampling())
    
    #os.system('clear')
    #printGridWorld(states_in_episode, actions_in_episode)
    #time.sleep(wait_time)
    
    return states_in_episode, actions_in_episode


# In[8]:

def generateGreedilySampledEpisode():
    current_state = START_STATE
    states_in_episode = []
    actions_in_episode = []
    while (current_state != END_STATE) & (len(states_in_episode) < MAXIMUM_EPISODE_LENGTH):
        states_in_episode.append(current_state)
        action = policy[current_state]
        actions_in_episode.append(action)
        
        #os.system('clear')
        #printGridWorld(states_in_episode, actions_in_episode)
        #time.sleep(wait_time)
        
        current_state = all_transitions.get(current_state).get(action)
    
    if current_state == END_STATE:
        states_in_episode.append(END_STATE)
        actions_in_episode.append(policy[END_STATE])
        
    #os.system('clear')
    #printGridWorld(states_in_episode, actions_in_episode)
    #time.sleep(wait_time)
        
    return states_in_episode, actions_in_episode


# In[9]:

for policy_iterator in range(MAXIMUM_NUMBER_OF_POLICY_ITERATIONS):
    print(RED("Policy iteration number " + str(policy_iterator) + "\n"))
    EPSILON = (1/((0.2 * policy_iterator) + 1))
    for episode_iterator in range(NUMBER_OF_EPISODES_PER_POLICY_EVALUATION):
        
        random_throw = random.uniform(0, 1)
        if random_throw < EPSILON:
            print("Generating samples ", BLUE("RANDOMLY"))
            states_in_episode, actions_in_episode = generateRandomlySampledEpisode()
        else:
            print("Generating samples ",BLUE("GREEDILY"))
            states_in_episode, actions_in_episode = generateGreedilySampledEpisode()
        
        print("Trajectory Generated", end = '')
        printGridWorld(states_in_episode, actions_in_episode)
        #print(states_in_episode)
        #print(actions_in_episode)
        
        for step_number in range(len(states_in_episode)):
            current_state = states_in_episode[step_number]
            if current_state == END_STATE:
                break
            current_action = actions_in_episode[step_number]
            
            total_state_action_visits[current_state][current_action] = total_state_action_visits[current_state][current_action] + 1
            estimated_discounted_return = stateActionBasedDiscountedReturn(states_in_episode[step_number:], actions_in_episode[step_number:])
            all_state_action_value_pairs[current_state][current_action] = all_state_action_value_pairs[current_state][current_action] + (1/total_state_action_visits[current_state][current_action]) * (estimated_discounted_return - all_state_action_value_pairs[current_state][current_action])    
    
    updatePolicy()
    #printStateActionValuePairs()
    printPolicy()
    print("\n")
    time.sleep(wait_time)


# In[ ]:



