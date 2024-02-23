import click
import joblib
import pandas as pd
from train import clean_data



@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
def predict(input_dataset, output_dataset):
    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    artifacts = joblib.load("models/artifacts.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    clean_data = artifacts["clean_data"] 
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Clean the data
    data = clean_data(data)

    # Extract the used data
    data = data[num_features + fl_features + cat_features]

    # Apply encoder on categorical features
    data_cat = enc.transform(data[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
            [
                data[num_features + fl_features].reset_index(drop=True),
                pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
            ],
            axis=1,
        )

    # Make predictions
    predictions = model.predict(data)

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}")
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # python predict.py -i "data/input.csv" -o "output/test.csv"
    predict()
