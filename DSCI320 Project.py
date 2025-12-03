# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# Set the path to the file you'd like to load
file_path = "nba_team_stats_00_to_23.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mharvnek/nba-team-stats-00-to-18",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

df.set_index(['Team','season'], inplace=True)


features = ['assists', 'turnovers', 'rebounds', 'field_goal_percentage', 'three_point_percentage']
df = df.dropna(subset=features + ["win_percentage"])
x = df[features].values
y = df["win_percentage"].values

print(df.head())
