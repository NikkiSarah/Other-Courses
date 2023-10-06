import numpy as np
import pandas as pd

# 1. What is the version of NumPy that you installed?
np_version = np.__version__
np_version

# 2. How many rows are in the dataset
price_data = pd.read_csv("car_price_data.csv")
num_rows = price_data.shape[0]
num_rows

# 3. Who are the top-three manufacturers?
manuf_counts = price_data.groupby('Make').size()
top_three = manuf_counts.sort_values(ascending=False)[:3]
top_three

# 4. How many unique Audi car models are there?
audi_cars = price_data.loc[price_data.Make == "Audi"]
num_models = audi_cars.Model.nunique()
num_models

# 5. How many columns in the dataset have missing values?
missing_values = price_data.isnull().sum()
num_missing_cols = len(missing_values[missing_values > 0])
num_missing_cols

# 6a. What is the median value of "Engine Cylinders"
median_cylinders = price_data["Engine Cylinders"].median()
median_cylinders

# 6b. What is the mode value of "Engine Cylinders"
mode_cylinders = price_data["Engine Cylinders"].mode()[0]
mode_cylinders

# 6c. Use fillna to fill the missing values in "Engine Cylinders" with the mode
price_data["Engine Cylinders"] = price_data["Engine Cylinders"].fillna(value=mode_cylinders)

# 6d. Calculate the median again
median_cylinders2 = price_data["Engine Cylinders"].median()
median_cylinders2

# 7a. Select all the "Lotus" cars
lotus_cars = price_data.loc[price_data.Make == "Lotus"]

# 7b. Select only "Engine HP" and "Engine Cylinders"
lotus_cars2 = lotus_cars[["Engine HP", "Engine Cylinders"]]

# 7c. Drop all duplicate rows
lotus_cars2_nodup = lotus_cars2.drop_duplicates()

# 7d. Get the underlying numpy array and call it X
X = np.array(lotus_cars2_nodup)

# 7e. Compute the matrix-matrix multiplication of the transpose of X and X. Call it XTX.
XTX = X.T @ X

# 7f. Compute the inverse of XTX
XTX_inv = np.linalg.inv(XTX)

# 7g. Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

# 7h. Multiply the inverse of XTX with the transpose of X and the multiply the result by y. Call the result w.
w = XTX_inv @ X.T @ y
first_element = w[0]
first_element
