# Model Detail: Predicting Belgian Property Prices

## Introduction
In this machine learning project, I aimed to create a supervised machine learning algorithm capable of predicting property prices accros Belgium based on multiple features.

## Dataset
__Source__: The input dataset was meticulously scraped from the Belgian real estate website **Immoweb**. I obtained explicit authorization from the estate company before collecting any data.

**Size**: The dataset contained multiple thousands of properties, totaling approximately 75,000 records.

**Property Types**: To ensure diversity, I included both houses and apartments. This balanced distribution across property types allowed my model to generalize effectively.

**Features**:
**Price**: The target variable I aimed to predict.

**Total living Square Meter**: An essential feature representing the size of the property.

**Energy Consumption**: A crucial factor influencing property value.

**Number of Bedrooms**: Reflects the property’s capacity.

**Coordinates**: Geographic information that could impact prices based on location.

**And more**: The dataset encompassed additional relevant features.

## Data Preprocessing

Before training the model, I performed thorough data preprocessing:

**Cleaning**: Addressed missing values, outliers, and inconsistencies.

**Feature Engineering**: Transformed existing features into more useful insights.

**Encoding**: Converted categorical variables into numerical representations.

## Model selection

Choosing an appropriate model is a very delicate, a bit how you would educate your child :). 
I explored various regression algorithms, as:

**Linear Regression**: A simple yet interpretable choice.

**Random Forest Regression**: Robust and capable of handling non-linear relationships.

**Gradient Boosting Regression**: Powerful for ensemble learning.

After many tests I sticked with Gradient Boosting Regression, albeit it's more prone to overfitting compared to the other models I've tested. The reason being, is because I got better results overall from stock and had to do little as of parameter tuning to achieve great results. 

## Performance of models

| Model         | Train R² Score | Test R² Score |
|---------------|----------------|---------------|
| LinearReg     | 0.26698        | 0.29827       |
| Random Forest | 0.94807        | 0.72913       |
| XGBoost       | 0.97382        | 0.80052       |

## Limitations

While my model isn't completly perfect is slightly overfitted, it is nonetheless capable of providing remarkable results.

## Usage

### **1. Install Dependencies**: To download the dependencies listed in the requirements.txt file, you can use the following command in your terminal or command prompt:

```pip install -r requirements.txt```

### **2. Generate Predictions**:
- Locate the predict.py script.
- Specify your specific input file (replace "data/input.csv" with your actual file path).
- Run the following command:
  
  ```python predict.py -i "data/input.csv" -o "output/test.csv"```
  
  You can edit the name for the output file as desired.

## Maintainers

If you have any questions regarding my model, feel free to reach out to me through the following social media channels:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/120px-Instagram_icon.png" alt="Instagram Logo" width="20"/>   [mimoun.atmani.18](https://www.instagram.com/mimoun.atmani.18/)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Gmail_icon_%282020%29.svg/120px-Gmail_icon_%282020%29.svg.png" alt="Email Logo" width="20"/> mimounb1597@gmail.com

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/120px-LinkedIn_logo_initials.png" alt="LinkedIn Logo" width="20"/> www.linkedin.com/in/mimoun-atmani
