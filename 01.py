import pandas as pd

# load the dataset
music_data = pd.read_csv('musicdata.csv')
print(music_data.head())

music_data.info()
print(music_data.describe())