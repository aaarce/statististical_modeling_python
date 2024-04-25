import pandas as pd

# load the dataset
music_data = pd.read_csv('musicdata.csv')
print(music_data.head())

music_data.info()
print(music_data.describe())
music_data.isnull().sum()
# dropping the 'Unnamed: 0' column
music_data_cleaned = music_data.drop(columns=['Unnamed: 0'])

# handling missing values by filling them with placeholder text
columns_with_missing_values = ['Track Name', 'Artists', 'Album Name']
music_data_cleaned[columns_with_missing_values] = music_data_cleaned[columns_with_missing_values].fillna('Unknown')