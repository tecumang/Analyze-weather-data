#if your dataset as no weather condition then follow this code
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
data = pd.read_csv('your datasdet.csv')

# Add a hypothetical 'WeatherCondition' column with random values
data['WeatherCondition'] = np.random.choice(['Sunny', 'Rainy'], size=len(data))

# Print the DataFrame to verify the changes
print(data)
#add this weathercodition in your dataset