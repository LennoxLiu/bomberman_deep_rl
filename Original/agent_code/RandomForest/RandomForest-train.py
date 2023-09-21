import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

if __name__ == "__main__":
    with open('./agent_code/RandomForest/train_data.pickle', 'rb') as file:
        observations, target_actions, rewards = pickle.load(file)

    model = RandomForestClassifier(n_estimators = 1000, ccp_alpha = 0.0001, n_jobs = -1, oob_score=True)
    metadata = {"global_steps": 0,"params": model.get_params()}
    #500
    n_splits = 5

    # Initialize the KFold object
    kf = KFold(n_splits=n_splits)

    # Initialize a variable to keep track of the performance metrics (e.g., accuracy)
    total_accuracy = 0

    # Iterate through the folds
    for train_index, test_index in tqdm(kf.split(target_actions)):
        X_train, X_test = observations[train_index], observations[test_index]
        y_train, y_test = target_actions[train_index], target_actions[test_index]
        weights_train, weights_test = rewards[train_index], rewards[test_index]

        total_rewards_train = sum([reward for reward in weights_train if reward > 0])
        weights_train /= total_rewards_train

        total_rewards_test =  sum([reward for reward in weights_test if reward > 0])
        weights_test /= total_rewards_test

        # Train the model
        # model.fit(X_train, y_train, sample_weight = weights_train)  # Use sample weights
        model.fit(X_train, y_train) # train without weight

        # Evaluate the model
        accuracy = model.score(X_test, y_test, sample_weight=weights_test)  # Use sample weights
        total_accuracy += accuracy

    # Calculate the average performance metric
    average_accuracy = total_accuracy / n_splits

    # Store the model
    joblib.dump(model, './agent_code/RandomForest/models/random_forest_model.joblib')
    with open('./agent_code/RandomForest/metadata.pickle', 'wb') as file:
        pickle.dump(metadata, file)

    metadata["global_steps"] += len(target_actions)*(n_splits - 1) # update global_steps
    
    print("\naverage_accuracy:", average_accuracy)
    print("OOB_score:", model.oob_score_)
    print("total_steps:",metadata["global_steps"])  