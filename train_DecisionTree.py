import os
import pickle
import shutil

from sklearn.ensemble import RandomForestClassifier
from CustomEnv import CustomEnv
from GetFeatures import GetFeatures
from RuleBasedAgent import RuleBasedAgent
from CustomEnv import ACTION_MAP
import settings as s
import torch as th
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count

# Crop the observation for smaller feature space
def crop_observation(observation, crop_size_1, crop_size_2):
    obs1, obs2 = observation[0], observation[1]
        
    crop_diam_1 = int((crop_size_1 - 1) / 2)
    crop_diam_2 = int((crop_size_2 - 1) / 2)
        
    # crop obs to crop_size x crop_size
    obs1 = obs1[s.ROWS-crop_diam_1:s.ROWS+crop_diam_1+1,s.COLS-crop_diam_1:s.COLS+crop_diam_1+1]
    obs2 = obs2[s.ROWS-crop_diam_2:s.ROWS+crop_diam_2+1,s.COLS-crop_diam_2:s.COLS+crop_diam_2+1]
    
    # flatten the cropped observation
    obs1 = obs1.reshape(crop_size_1* crop_size_1)
    obs2 = obs2.reshape(crop_size_2* crop_size_2)

    # Reshape and standardize the input to [0,1]
    # May or may not need to standardize the input for decision tree
    obs1 = obs1 / 8
    obs2 = obs2 / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

    return np.concatenate((obs1, obs2))

def generate_data(n_rounds, crop_size_1, crop_size_2):
    # Create a custom environment, can configure the opponents through options
    option_rule =  {"argv": ["play","--no-gui","--my-agent","user_agent", "--train", "1"]}
    option_random =  {"argv": ["play","--no-gui","--agents","user_agent", "random_agent", "random_agent", "random_agent", "--train", "1"]}
    
    env = CustomEnv(options= option_random)
    agent = RuleBasedAgent(has_memory=True) # mimic the rule-based agent
    feature_extractor = GetFeatures()

    observations = []
    actions = []
    # can do faster by multiprocessing
    for _ in tqdm(range(n_rounds)):
        observation, game_state = env.reset()
        feature_extractor.reset()

        observation_crop = crop_observation(observation, crop_size_1, crop_size_2)
        features = feature_extractor.state_to_features(game_state)
        observations.append(np.concatenate((features, observation_crop)))
        
        agent.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.act(game_state) # act based on un-cropped observation
            action_index = ACTION_MAP.index(action)
            actions.append(action_index)

            observation, reward, terminated, truncated, game_state = env.step(action_index)
            observation_crop = crop_observation(observation, crop_size_1, crop_size_2)
            features = feature_extractor.state_to_features(game_state)
        
            if terminated or truncated:
                break
            else:
                observations.append(np.concatenate((features, observation_crop)))

    return np.array(observations), np.array(actions)

def prepare_data(n_rounds, crop_size_1, crop_size_2):
    num_workers = cpu_count()
    rounds_per_worker = n_rounds // num_workers
    pool = Pool(num_workers)
    
    results = pool.starmap(generate_data, [(rounds_per_worker, crop_size_1, crop_size_2) for _ in range(num_workers)])
    
    pool.close()
    pool.join()
    
    observations = np.concatenate([result[0] for result in results], axis=0)
    actions = np.concatenate([result[1] for result in results], axis=0)
    
    return observations, actions


def train_decision_tree(observations, actions):
        # Initialize and train the decision tree classifier
        # clf = DecisionTreeClassifier()
        clf = RandomForestClassifier(n_estimators=20, max_features=0.99)
        clf.fit(observations, actions)
        
        return clf

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('decision_tree', exist_ok=True)
    os.makedirs('Original/agent_code/DecisionTree_agent', exist_ok=True)
    # Save the code for reference
    shutil.copyfile('CustomEnv.py', 'decision_tree/CustomEnv.py')
    shutil.copyfile('train_utils.py', 'decision_tree/train_utils.py')
    shutil.copyfile('train_DecisionTree.py', 'decision_tree/train_DecisionTree.py')
    shutil.copyfile('GetFeatures.py', 'decision_tree/GetFeatures.py')
    
    save_to_agent_folder = input(
    "Do you want to replace the files in agent_code folder with current setup? (y/n): ")
    if save_to_agent_folder.lower() == 'y':
        shutil.copyfile('CustomEnv.py', 'Original/agent_code/DecisionTree_agent/CustomEnv.py')
        shutil.copyfile('train_utils.py', 'Original/agent_code/DecisionTree_agent/train_utils.py')
        shutil.copyfile('train_DecisionTree.py', 'Original/agent_code/DecisionTree_agent/train_DecisionTree.py')
        shutil.copyfile('GetFeatures.py', 'Original/agent_code/DecisionTree_agent/GetFeatures.py')

    n_rounds = 2000  # Number of rounds to generate data
    crop_size_1 = 9  # crop size for field map
    crop_size_2 = 17  # crop size for bomb map
    
    # observations, actions = prepare_data(n_rounds, crop_size_1, crop_size_2)
    # load the data from the npy files
    observations = np.load('decision_tree/observations.npy')
    actions = np.load('decision_tree/actions.npy')

    os.makedirs('decision_tree', exist_ok=True)
    
    # Store the data in a pickle file
    # np.save('decision_tree/observations.npy', observations)
    # np.save('decision_tree/actions.npy', actions)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(observations, actions, test_size=0.1, random_state=42)
    
    # Train the decision tree on the training set
    decision_tree = train_decision_tree(X_train, y_train)
    
    # Validate the model on the training set
    y_train_pred = decision_tree.predict(X_train[:100])
    train_accuracy = accuracy_score(y_train[:100], y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.3f}")

    # Validate the model on the validation set
    y_pred = decision_tree.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.3f}")
    
    # Save the trained model
    with open('decision_tree/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(decision_tree, f)
    
    if save_to_agent_folder.lower() == 'y':
        shutil.copyfile('decision_tree/decision_tree_model.pkl', 'Original/agent_code/DecisionTree_agent/decision_tree_model.pkl')
        print("Model saved to agent_code folder")