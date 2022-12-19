import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression , RidgeCV, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns

################################### Function ###################################
def assign_mlb_rpg(year):
    return avg_runs_per_game[year]

########################## Fetching Data From Database ##########################

# Connecting to sqlite database
conn = sqlite3.connect("lahman2016.sqlite")

# Querying database for all seasons where a team played 150 or more games and is still active today
query = """
SELECT
    *
FROM
    Teams
INNER JOIN
    TeamsFranchises
    ON
        Teams.franchID = TeamsFranchises.franchID

    WHERE
        Teams.G >= 150 and TeamsFranchises.active = "Y";
"""

# Creating dataframe from query
Teams = conn.execute(query).fetchall()
# Convert "teams" to dataframe
teams_df = pd.DataFrame(Teams)

######################## Cleaning and Preparing The Data ########################

# Adding column names to dataframe
cols = ["yearID", "lgID", "teamID", "franchID", "divID", "Rank", "G", "GHome",
"W", "L", "DivWin", "WCWin", "LgWin", "WSWin", "R", "AB", "H", "2B", "3B", "HR",
"BB", "SO", "SB", "CS", "HBP", "SF", "RA", "ER", "ERA", "CG", "SHO", "SV", "IPouts",
"HA", "HRA", "BBA", "SOA", "E", "DP", "FP", "name", "park", "attendance", "BPF",
"PPF", "teamIDBR", "teamIDlahan45", "teamIDretro", "franchID", "franchName",
"active", "NAassoc"]
teams_df.columns = cols

# Eliminating the columns that aren't necessary or derived from the win column
drop_cols = ["lgID", "franchID", "divID", "Rank", "GHome", "L", "DivWin", "WCWin",
"LgWin", "WSWin", "SF", "name", "park", "attendance", "BPF", "PPF", "teamIDBR",
"teamIDlahan45", "teamIDretro", "franchID", "franchName", "active", "NAassoc"]
df = teams_df.drop(drop_cols, axis=1)

# Dealing with null values of columns
isnull = df.isnull().sum(axis=0).tolist()

# Eliminating columns with large amounts of null values
df = df.drop(["CS", "HBP"], axis=1)

# Filling null values with the median values of the columns
df["SO"] = df["SO"].fillna(df["SO"].median())
df["DP"] = df["DP"].fillna(df["DP"].median())

###################### Exploring and Visualizing The Data ######################

# Plotting distribution of wins
plt.hist(df["W"])
plt.xlabel("Wins")
plt.title("Distribution of Wins")
plt.show()

# Plot scatter graph of Year vs. Wins
plt.scatter(df["yearID"], df["W"])
plt.title("Wins Scatter Plot")
plt.xlabel("Year")
plt.ylabel("Wins")
plt.show()

# Calculate average runs per game
runs_per_year, games_per_year = {}, {}
for i,row in df.iterrows():
    year, runs, games = row["yearID"], row["R"], row["G"]
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games
avg_runs_per_game = {}
for year,runs in runs_per_year.items():
    games = games_per_year[year]
    avg_runs_per_game[year] = runs / games

lists = sorted(avg_runs_per_game.items())
x, y = zip(*lists)

# Plot MLB yearly runs per game
plt.plot(x, y)
plt.title("MLB Yearly Runs Per Game")
plt.xlabel("Year")
plt.ylabel("MLB Runs Per Game")
plt.show()

############################## Adding New Features ##############################

# MLB average runs per game
df["mlb_rpg"] = df["yearID"].apply(assign_mlb_rpg)

# Runs per game
df["R_per_game"] = df["R"] / df["G"]

# Runs allowed per game
df["RA_per_game"] = df["RA"] / df["G"]

# Run differential
df["R_diff"] = df["R"] - df["RA"]

# Scatter plots
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# Scatter plot of runs per game vs. wins
ax1.scatter(df["R_per_game"], df["W"], c="blue")
ax1.set_title("Runs per Game vs. Wins")
ax1.set_xlabel("Runs per Game")
ax1.set_ylabel("Wins")

# Scatter plot of runs allowed per game vs. wins
ax2.scatter(df["RA_per_game"], df["W"], c="red")
ax2.set_title("Runs Allowed per Game vs. Wins")
ax2.set_xlabel("Runs Allowed per Game")
ax2.set_ylabel("Wins")
plt.show()

# Determine and display how each variable/feature correlates with each other
corr = df.corr(numeric_only=True)
plt.subplots(figsize=(18,10.5))
sns.heatmap(corr, annot = True)
plt.show()

# Determine and display how each variable/feature correlates with the target variable
w_corr = corr[["W"]]
plt.subplots(figsize=(10,9))
sns.heatmap(w_corr, annot = True)
plt.show()

################################ Building Model ################################

# Numerical columns needed for regression algorithms
numeric_cols = ["G", "R", "AB", "H", "2B", "3B", "HR", "BB", "SO", "SB", "RA", "ER",
"ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA", "SOA", "E", "DP", "FP",
"R_per_game", "RA_per_game", "mlb_rpg", "W"]

# Numerical columns without the target variable (wins)
attributes = ["G", "R", "AB", "H", "2B", "3B", "HR", "BB", "SO", "SB", "RA", "ER",
"ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA", "SOA", "E", "DP", "FP",
"R_per_game", "RA_per_game", "mlb_rpg"]
data_attributes = df[attributes]

data = df[numeric_cols]
x = data[attributes]
y = data["W"]

# Split the train and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.25, shuffle = True
)

# Linear Regression
classifier = LinearRegression()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

# Calculate linear regression mean absoulte error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error of Linear Regression:", mae)

# Plot linear regression results
plt.subplots(figsize=(16, 9))
plt.scatter(x_test["R_per_game"], y_test)
plt.scatter(x_test["R_per_game"], predictions)
plt.title("Linear Regression Win Predictions")
plt.xlabel("Runs Per Game")
plt.ylabel("Wins")
plt.legend(["True Wins","Predicted Wins"])
plt.show()

# Ridge CV
classifier =  RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)

# Calculate ridge cv mean absoulte error
mae = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error of Ridge CV:", mae)

# Plot ridge cv results
plt.subplots(figsize=(16, 9))
plt.scatter(x_test["R_per_game"], y_test)
plt.scatter(x_test["R_per_game"], predictions)
plt.title("Ridge CV Win Predictions")
plt.xlabel("Runs Per Game")
plt.ylabel("Wins")
plt.legend(["True Wins","Predicted Wins"])
plt.show()

# Lasso
classifier =  Lasso(alpha=0.1)
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)

# Calculate lasso mean absoulte error
mae = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error of Lasso:", mae)

# Plot lasso results
plt.subplots(figsize=(16, 9))
plt.scatter(x_test["R_per_game"], y_test)
plt.scatter(x_test["R_per_game"], predictions)
plt.title("Lasso Win Predictions")
plt.xlabel("Runs Per Game")
plt.ylabel("Wins")
plt.legend(["True Wins","Predicted Wins"])
plt.show()
