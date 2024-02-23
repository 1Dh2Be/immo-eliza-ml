import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV




def clean_data(data):
    """Clean the dataset"""
    # Impute missing values for total_area_sqm based on property_type and subproperty_type
    mean_sqm_per_category = data.groupby(['property_type', 'subproperty_type'])['total_area_sqm'].median()

    # Fill missing values in 'total_area_sqm' based on median values per category
    data['total_area_sqm'] = data.apply(
        lambda row: mean_sqm_per_category.loc[(row['property_type'], row['subproperty_type'])] 
                     if pd.isna(row['total_area_sqm']) 
                     else row['total_area_sqm'],
        axis=1
    )
    
    # Impute missing values for primary_energy_consumption_sqm based on property_type and province
    median_energy_consumption_sqm = data.groupby(['property_type', 'province'])['primary_energy_consumption_sqm'].median()

    # Fill missing values in 'primary_energy_consumption_sqm' based on median values per category
    data['primary_energy_consumption_sqm'] = data.apply(
        lambda row: median_energy_consumption_sqm.loc[(row['property_type'], row['province'])] 
                     if pd.isna(row['primary_energy_consumption_sqm']) 
                     else row['primary_energy_consumption_sqm'],
        axis=1
    )

    # Delete the rows where there is no province
    data = data[data["province"] != "MISSING"]
    
    # Drop the rows where 'primary_energy_consumption_sqm' is over 1000
    data = data[data['primary_energy_consumption_sqm'] <= 1000]

    # Drop the rows where there are more than 105 bedrooms
    data = data[data['nbr_bedrooms'] <= 50]
    
    return data

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    #  Clean the data
    data = clean_data(data)

    # Define features to use
    num_features = ["total_area_sqm", "nbr_bedrooms", "primary_energy_consumption_sqm", "latitude", "longitude", "terrace_sqm"]
    fl_features = ["fl_terrace", "fl_garden"]
    cat_features = ["subproperty_type", "province", "state_building", "property_type"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Define the parameter grid
    param_grid = {
        'n_estimators': [650],
        'max_depth': [7],
        'learning_rate': [0.075],
    }

    # Initialize the XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=505)

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(best_params)

    # Train the model with the best parameters
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=505, **best_params)
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}", end="")
    print(f"Test R² score: {test_score}")

    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "clean_data": clean_data,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")

if __name__ == "__main__":
    train()
