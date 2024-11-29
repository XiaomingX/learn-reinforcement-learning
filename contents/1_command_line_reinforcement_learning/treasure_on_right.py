import numpy as np
import pandas as pd
import time

# Setting random seed for reproducibility
np.random.seed(2)

# Constants
N_STATES = 6  # The length of the 1-dimensional world
ACTIONS = ['left', 'right']  # Available actions
EPSILON = 0.9  # Exploration rate for epsilon-greedy policy
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
MAX_EPISODES = 13  # Maximum number of episodes to run
FRESH_TIME = 0.3  # Time delay for each move to visualize the process

# Function to initialize the Q-table
def build_q_table(n_states, actions):
    """
    Initializes the Q-table with zeros.
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # Initialize Q-values to zero
        columns=actions  # Actions are column names
    )
    return table

# Function to choose an action based on the epsilon-greedy policy
def choose_action(state, q_table):
    """
    Chooses an action based on the current state and the Q-table using epsilon-greedy strategy.
    """
    state_actions = q_table.iloc[state, :]  # Get all actions for the current state
    if np.random.uniform() > EPSILON or (state_actions == 0).all():  # Random action if exploring
        action_name = np.random.choice(ACTIONS)
    else:  # Exploit: choose the action with the highest Q-value
        action_name = state_actions.idxmax()  # idxmax gives the column name (action) with the max value
    return action_name

# Function to interact with the environment
def get_env_feedback(state, action):
    """
    Simulates the environment's response to the agent's action.
    """
    if action == 'right':  # Moving right
        if state == N_STATES - 2:  # Reached the goal (treasure)
            return 'terminal', 1  # Reward of 1 for reaching the goal
        else:
            return state + 1, 0  # Otherwise, move one step right with no reward
    else:  # Moving left
        if state == 0:  # Hit the wall, cannot move left
            return state, 0  # No reward and no state change
        else:
            return state - 1, 0  # Move one step left with no reward

# Function to update and print the environment state
def update_env(state, episode, step_counter):
    """
    Updates the environment's visual state.
    """
    env_list = ['-'] * (N_STATES - 1) + ['T']  # Environment setup with the treasure at the last position
    if state == 'terminal':  # If the agent reaches the goal, print the results
        print(f'\rEpisode {episode + 1}: total_steps = {step_counter}', end='')
        time.sleep(2)  # Pause before clearing the screen
        print('\r' + ' ' * 30, end='')  # Clear the line
    else:
        env_list[state] = 'o'  # Place the agent 'o' in the environment
        interaction = ''.join(env_list)  # Create a string representation of the environment
        print(f'\r{interaction}', end='')  # Print the environment on the same line
        time.sleep(FRESH_TIME)  # Wait for a brief moment to show the agent's movement

# The main Q-learning function
def rl():
    """
    Main function to perform Q-learning and train the agent.
    """
    # Initialize the Q-table
    q_table = build_q_table(N_STATES, ACTIONS)

    # Loop through episodes
    for episode in range(MAX_EPISODES):
        state = 0  # Start at the beginning of the world
        step_counter = 0
        is_terminated = False
        update_env(state, episode, step_counter)  # Update the environment before the loop starts

        # Loop through steps until the episode ends
        while not is_terminated:
            action = choose_action(state, q_table)  # Choose an action
            next_state, reward = get_env_feedback(state, action)  # Get feedback from the environment

            # Predict Q-value for the current state-action pair
            q_predict = q_table.loc[state, action]

            if next_state != 'terminal':  # If the next state is not terminal, calculate Q-target
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()  # Max Q-value for next state
            else:  # If it's the terminal state, the target is just the reward
                q_target = reward
                is_terminated = True  # End the episode

            # Update the Q-value for the state-action pair
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)

            state = next_state  # Move to the next state
            update_env(state, episode, step_counter + 1)  # Update the environment
            step_counter += 1

    return q_table  # Return the learned Q-table

# Main function to tie everything together
def main():
    """
    Main function to execute the Q-learning process and print the results.
    """
    print("Starting the Q-learning process...")
    q_table = rl()  # Run the Q-learning algorithm
    print('\nQ-table after training:\n')
    print(q_table)  # Display the final Q-table

if __name__ == "__main__":
    main()  # Run the program
