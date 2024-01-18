"""a)"""

import matplotlib.pyplot as plt
import pandas as pd

# Reload the CSV file with proper headers
file_path = 'C:/Users/Vladth01/OneDrive - unibuc.ro/An IV/Semestrul I/Procesarea semnalelor/Saptamana_10/Lab_10/co2_daily_mlo.csv'
column_names = ['Year', 'Month', 'Day', 'DecimalDate', 'CO2']
data = pd.read_csv(file_path, names=column_names)

# Group by Year and Month, and calculate the monthly average of CO2 emissions
monthly_avg_co2 = data.groupby(['Year', 'Month'])['CO2'].mean().reset_index()

# Creating a simpler plot by reducing the number of data points
# For example, we can plot only the first month of each year
simplified_data = monthly_avg_co2[monthly_avg_co2['Month'] == 1]
plt.figure(figsize=(10, 5))
plt.plot(simplified_data['Year'], simplified_data['CO2'], marker='o')
plt.title('January Average CO2 Levels Over the Years')
plt.xlabel('Year')
plt.ylabel('Average CO2 (ppm)')
plt.grid(True)
plt.tight_layout()
plt.show()

"""b)"""
from sklearn.linear_model import LinearRegression
# Prepare the data for linear regression model
# We use the year and month as our features. Since we're doing a simple linear regression, 
# we'll convert the year and month into a single numerical feature.
monthly_avg_co2['YearMonthNumeric'] = monthly_avg_co2['Year'] + monthly_avg_co2['Month']/12


# Reshaping the data for sklearn
X = monthly_avg_co2['YearMonthNumeric'].values.reshape(-1, 1)
y = monthly_avg_co2['CO2'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict the CO2 values
monthly_avg_co2['Predicted_CO2'] = model.predict(X)

# Calculate the residuals (actual - predicted)
monthly_avg_co2['Residuals'] = monthly_avg_co2['CO2'] - monthly_avg_co2['Predicted_CO2']
monthly_avg_co2.head()

# Plotting the original data, the trend line, and the detrended data
plt.figure(figsize=(15, 7))

# Plotting the original CO2 data
plt.plot(monthly_avg_co2['YearMonthNumeric'], monthly_avg_co2['CO2'], label='Original Data', color='blue', alpha=0.5)

# Plotting the trend line
plt.plot(monthly_avg_co2['YearMonthNumeric'], monthly_avg_co2['Predicted_CO2'], label='Trend Line', color='red')

# Plotting the detrended data
plt.plot(monthly_avg_co2['YearMonthNumeric'], monthly_avg_co2['Residuals'], label='Detrended Data', color='green', alpha=0.5)
plt.title('CO2 Levels: Original, Trend, and Detrended Data')
plt.xlabel('Year-Month')
plt.ylabel('CO2 Levels (ppm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""c)"""
import numpy as np
# Selecting the last 12 months of data from the detrended dataset
last_12_months = monthly_avg_co2.sort_values(by='YearMonthNumeric').tail(12)

# Preparing the training data (X) and observations (y)
X_train = last_12_months['YearMonthNumeric'].values.reshape(-1, 1)
y_train = last_12_months['Residuals'].values

# Define a simple kernel function. Here, we use the Radial Basis Function (RBF) kernel.
def rbf_kernel(x1, x2, length_scale):
    """ Radial Basis Function (RBF) kernel """
    return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / length_scale**2)

# Length scale for the RBF kernel
length_scale = 1.0

# Constructing the covariance matrix (K) for the training data
K = np.zeros((len(X_train), len(X_train)))
for i in range(len(X_train)):

    for j in range(len(X_train)):

        K[i, j] = rbf_kernel(X_train[i], X_train[j], length_scale)
        
# Adding noise variance (sigma^2 * I) to the covariance matrix
sigma_squared = 0.1  # Assumed noise variance
K += sigma_squared * np.eye(len(X_train))

# Invert the covariance matrix
K_inv = np.linalg.inv(K)

# Predictions for the same points (training set)
mu = np.dot(K_inv, y_train)

# Plotting the training data and the prediction from Gaussian Process
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, mu, color='red', label='GP Prediction')
plt.title('Gaussian Process Regression on Detrended CO2 Data')
plt.xlabel('Year-Month')
plt.ylabel('CO2 Levels (Detrended)')
plt.legend()
plt.grid(True)
plt.show()