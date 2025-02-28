import pandas as pd
music_data = pd.read_csv('music.csv')

x = music_data.drop(columns=['genre']) #features
y = music_data['genre'] #labels

print(y)