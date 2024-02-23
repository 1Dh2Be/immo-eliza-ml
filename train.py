import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
    
    return data

def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    #  Clean the data
    data = clean_data(data)

    # Define features to use
    prop_feature = "property_type"
    province_feature = 'province'
    subprop_feature = 'subproperty_type'
    sqm_feature = 'total_area_sqm'
    bedrooms_feature = 'nbr_bedrooms'
    terrace_feature = 'fl_terrace'
    garden_feature = 'fl_garden'
    building_state = 'state_building'
    energy_cons_feature = 'primary_energy_consumption_sqm'

    features = [prop_feature, province_feature, subprop_feature, sqm_feature, bedrooms_feature, 
            terrace_feature, garden_feature, building_state, energy_cons_feature]

    # Split the data into features and target
    X = data[features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    # median = {}
    # for column in ['total_area_sqm', 'primary_energy_consumption_sqm']:
    #     median[column] = X_train.groupby(['property_type', 'province'])[column].median()

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    columns_to_encode = ["subproperty_type", "province", "state_building", "property_type"]

    enc = OneHotEncoder()
    enc.fit(X_train[columns_to_encode])
    X_train_cat = enc.transform(X_train[columns_to_encode]).toarray()
    X_test_cat = enc.transform(X_test[columns_to_encode]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    num_features = list(set(features) - set(columns_to_encode))  # assuming all other features are numerical
    X_train = pd.concat(
        [
            X_train[num_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out(input_features=columns_to_encode)),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out(input_features=columns_to_encode)),
        ],
        axis=1,
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    artifacts = {
        "features": features,
        "enc": enc,
        "model": model,
        "clean_data": clean_data
    }
    joblib.dump(artifacts, "models/artifacts.joblib")

if __name__ == "__main__":
    train()
