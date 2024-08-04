import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return None

def normalize_features(data, features, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_features = scaler.fit_transform(data[features])
    scaled_df = pd.DataFrame(scaled_features, columns=features)
    return scaled_df, scaler

def save_results(data, filepath):
    data.to_csv(filepath, index=False)

def plot_predictions(y_test_valence, y_pred_valence, y_test_energy, y_pred_energy):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_valence, y_pred_valence, alpha=0.5)
    plt.plot([y_test_valence.min(), y_test_valence.max()], [y_test_valence.min(), y_test_valence.max()], 'k--', lw=2)
    plt.xlabel('Actual Valence')
    plt.ylabel('Predicted Valence')
    plt.title('Actual vs Predicted Valence')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_energy, y_pred_energy, alpha=0.5)
    plt.plot([y_test_energy.min(), y_test_energy.max()], [y_test_energy.min(), y_test_energy.max()], 'k--', lw=2)
    plt.xlabel('Actual Energy')
    plt.ylabel('Predicted Energy')
    plt.title('Actual vs Predicted Energy')

    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

def plot_valence_energy_distribution(valence, energy):
    plt.figure(figsize=(10, 6))
    plt.scatter(valence, energy, alpha=0.5)
    plt.xlabel('Valence')
    plt.ylabel('Energy')
    plt.title('Distribution of Valence and Energy')
    plt.show()

if __name__ == "__main__":
    input_filepath = 'tracks_features.csv'
    valence_energy_filepath = 'tracks_info.csv'
    output_filepath = 'predicted_valence_energy_scores_rf.csv'

    data = load_data(input_filepath)
    valence_energy_data = load_data(valence_energy_filepath)
    
    if data is None or valence_energy_data is None:
        exit()

    if 'valence' not in valence_energy_data.columns or 'energy' not in valence_energy_data.columns or 'isrc' not in valence_energy_data.columns:
        print("Error: The 'valence', 'energy', or 'isrc' column is not present in the valence data.")
        exit()

    merged_data = pd.merge(data, valence_energy_data[['isrc', 'valence', 'energy']], on='isrc')

    features = merged_data.columns.difference(['isrc', 'valence', 'energy'])

    scaled_df, scaler = normalize_features(merged_data, features)

    X = scaled_df
    y_valence = merged_data['valence']
    y_energy = merged_data['energy']

    # Check for NaN values in the features and target
    if X.isna().any().any():
        print("NaN values found in features, removing rows with NaN values.")
        nan_indices = X.isna().any(axis=1)
        X = X.dropna()
        y_valence = y_valence[~nan_indices]
        y_energy = y_energy[~nan_indices]

    if y_valence.isna().any():
        print("NaN values found in valence, removing rows with NaN values.")
        nan_indices = y_valence.isna()
        y_valence = y_valence.dropna()
        X = X[~nan_indices]
        y_energy = y_energy[~nan_indices]

    if y_energy.isna().any():
        print("NaN values found in energy, removing rows with NaN values.")
        nan_indices = y_energy.isna()
        y_energy = y_energy.dropna()
        X = X[~nan_indices]
        y_valence = y_valence[~nan_indices]

    X_train, X_test, y_train_valence, y_test_valence, y_train_energy, y_test_energy = train_test_split(X, y_valence, y_energy, test_size=0.2, random_state=42)

    plot_valence_energy_distribution(y_train_valence, y_train_energy)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model_valence = RandomForestRegressor(random_state=42)
    grid_search_valence = GridSearchCV(estimator=rf_model_valence, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search_valence.fit(X_train, y_train_valence)

    best_params_valence = grid_search_valence.best_params_
    best_score_valence = grid_search_valence.best_score_
    print(f"Best parameters for valence: {best_params_valence}")
    print(f"Best cross-validation score for valence: {best_score_valence}")

    best_rf_model_valence = grid_search_valence.best_estimator_
    best_rf_model_valence.fit(X_train, y_train_valence)
    y_pred_valence = best_rf_model_valence.predict(X_test)

    mse_valence = mean_squared_error(y_test_valence, y_pred_valence)
    r2_valence = r2_score(y_test_valence, y_pred_valence)
    print(f"Tuned Random Forest for Valence - Mean Squared Error: {mse_valence}")
    print(f"Tuned Random Forest for Valence - R^2 Score: {r2_valence}")

    rf_model_energy = RandomForestRegressor(random_state=42)
    grid_search_energy = GridSearchCV(estimator=rf_model_energy, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search_energy.fit(X_train, y_train_energy)

    best_params_energy = grid_search_energy.best_params_
    best_score_energy = grid_search_energy.best_score_
    print(f"Best parameters for energy: {best_params_energy}")
    print(f"Best cross-validation score for energy: {best_score_energy}")

    best_rf_model_energy = grid_search_energy.best_estimator_
    best_rf_model_energy.fit(X_train, y_train_energy)
    y_pred_energy = best_rf_model_energy.predict(X_test)

    mse_energy = mean_squared_error(y_test_energy, y_pred_energy)
    r2_energy = r2_score(y_test_energy, y_pred_energy)
    print(f"Tuned Random Forest for Energy - Mean Squared Error: {mse_energy}")
    print(f"Tuned Random Forest for Energy - R^2 Score: {r2_energy}")

    plot_predictions(y_test_valence, y_pred_valence, y_test_energy, y_pred_energy)
    plot_feature_importances(best_rf_model_valence, features)
    plot_feature_importances(best_rf_model_energy, features)

    # Ensure that the length of the DataFrame and the predicted values are the same
    if len(X) != len(merged_data):
        print("Length mismatch between features and merged data. Adjusting the DataFrame.")
        merged_data = merged_data.loc[X.index]

    merged_data['predicted_valence_rf'] = best_rf_model_valence.predict(X)
    merged_data['predicted_energy_rf'] = best_rf_model_energy.predict(X)

    merged_data['predicted_valence_rf'] = (merged_data['predicted_valence_rf'] - merged_data['predicted_valence_rf'].min()) / (merged_data['predicted_valence_rf'].max() - merged_data['predicted_valence_rf'].min())
    merged_data['predicted_energy_rf'] = (merged_data['predicted_energy_rf'] - merged_data['predicted_energy_rf'].min()) / (merged_data['predicted_energy_rf'].max() - merged_data['predicted_energy_rf'].min())

    results_rf = pd.DataFrame(merged_data[['isrc', 'predicted_valence_rf', 'predicted_energy_rf']])
    print(results_rf)
    save_results(results_rf, output_filepath)