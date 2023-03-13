import random

import pandas as pd
import numpy as np
import hyperopt

from hyperopt import hp, fmin, Trials, tpe
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import brier_score_loss

from elo import *

compact_results_rs = pd.read_csv('data/MRegularSeasonCompactResults.csv')
compact_results_mm = pd.read_csv('data/MNCAATourneyCompactResults.csv')
detailed_results_rs = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
detailed_results_mm = pd.read_csv('data/MNCAATourneyDetailedResults.csv')
# Data Description:
#   WFGM - field goals made (by the winning team)
#   WFGA - field goals attempted (by the winning team)
#   WFGM3 - three pointers made (by the winning team)
#   WFGA3 - three pointers attempted (by the winning team)
#   WFTM - free throws made (by the winning team)
#   WFTA - free throws attempted (by the winning team)
#   WOR - offensive rebounds (pulled by the winning team)
#   WDR - defensive rebounds (pulled by the winning team)
#   WAst - assists (by the winning team)
#   WTO - turnovers committed (by the winning team)
#   WStl - steals (accomplished by the winning team)
#   WBlk - blocks (accomplished by the winning team)
#   WPF - personal fouls committed (by the winning team)
mm_seeds = pd.read_csv('data/MNCAATourneySeeds.csv')
mm_seeds['Seed'] = mm_seeds['Seed'].apply(lambda x: float(x[1:3]))


# Get all team IDs
teams = np.unique(
    compact_results_rs[['WTeamID', 'LTeamID']].to_numpy().flatten()
)

# Run the Elo Model
elo_space = {
    'end_of_season_decay': hp.uniform('end_of_season_decay', 0, 1),
    'K': hp.uniform('K', 10, 50),
    'K_multiplier': hp.uniform('K_multiplier', 1, 4),
    'K_decay': hp.uniform('K_decay', 0, 1),
    'home_court_advantage': hp.uniform('home_court_advantage', 30, 200),
    'margin_of_victor_multiplier': hp.uniform('margin_of_victor_multiplier', 0, 1),
}


def space_to_elo(space):
    return ELOModel(teams=teams, starting_rating=1200, **space)


def elo_objective(space):
    elo_model = space_to_elo(space)

    for year in range(1993, 2013):
        results = compact_results_rs[compact_results_rs['Season'] == year]
        simulate_games(elo_model, results)

        elo_model.time_decay()

    loss = 0
    count = 0

    for year in range(2013, 2023):
        # Same, but tourney counts towards predictive score

        results = compact_results_rs[compact_results_rs['Season'] == year]
        simulate_games(elo_model, results)

        tourny_results = compact_results_mm[compact_results_mm['Season'] == year]
        for _, game in tourny_results.iterrows():
            wid, lid = game['WTeamID'], game['LTeamID']

            prediction = elo_model.predict_game(wid, lid, home_team=None)
            loss += (1 - prediction) ** 2
            count += 1

        elo_model.time_decay()

    return {'loss': loss / count, 'status': hyperopt.STATUS_OK}


elo_trials = Trials()
elo_best_params = fmin(elo_objective, elo_space, algo=tpe.suggest, max_evals=100, trials=elo_trials)
# elo_best_params = {'K': 10.067338276694205, 'K_decay': 0.06949750095380997,
#                    'K_multiplier': 2.9002766872973336, 'end_of_season_decay': 0.051425562928002384,
#                    'home_court_advantage': 51.533632254163315, 'margin_of_victor_multiplier': 0.362499069221264}

print('ELO Minimization Complete')


# Elo serves as a phase one pass
# Next is a few steps of data preparation on the detailed dataset
seasonal_averages = {}
seasonal_weighted_averages = {}

for season in range(1985, 2024):
    games = detailed_results_rs[detailed_results_rs['Season'] == season]
    seasonal_averages[season] = {}
    seasonal_weighted_averages[season] = {}

    for team in teams:
        # Get any game in which the team won and lost, data is in different places for both
        team_wins = games[(games['WTeamID'] == team)]
        team_losses = games[(games['LTeamID'] == team)]

        team_stats = {}
        team_weighted_stats = {}

        for col in games.columns:
            # Only need to do each stat once
            if not col.startswith('W') or col in ('WTeamID', 'WLoc'):
                continue

            # Cut off the W
            col = col[1:]
            team_stats[col] = 0
            total = 0

            team_stats[col] += team_wins[f'W{col}'].sum()
            total += len(team_wins)

            team_stats[col] += team_losses[f'L{col}'].sum()
            total += len(team_losses)

            if total != 0:
                team_stats[col] /= total

        seasonal_averages[season][team] = team_stats
        seasonal_weighted_averages[season][team] = team_weighted_stats

# Next we can use these averages to build a dataset for the tournament
elo_model = space_to_elo(elo_best_params)
dataset = []

# Simulate until 2003 using same data
for season in range(1993, 2003):
    regular_season_games = compact_results_rs[compact_results_rs['Season'] == season]
    simulate_games(elo_model, regular_season_games)

# Now actually build the dataset
for season in range(2003, 2024):
    regular_season_games = compact_results_rs[compact_results_rs['Season'] == season]
    simulate_games(elo_model, regular_season_games)

    tournament_games = compact_results_mm[compact_results_mm['Season'] == season]
    tournament_seeds = mm_seeds[mm_seeds['Season'] == season]

    for _, game in tournament_games.iterrows():
        winner_stats = seasonal_averages[season][game['WTeamID']]
        loser_stats = seasonal_averages[season][game['LTeamID']]
        winner_seed = tournament_seeds[tournament_seeds['TeamID'] == game['WTeamID']]['Seed'].item()
        loser_seed = tournament_seeds[tournament_seeds['TeamID'] == game['LTeamID']]['Seed'].item()
        entry = {}

        # In each entry, season averages will be measured along with the elo of each team
        # Ideally, I would also like to include distance to the game, but that is quite
        # difficult (I do not have that data)
        # Note: We need to randomly decide because otherwise the y column has no variation
        if random.random() < 0.5:
            # Winner gets slot 1
            for key in winner_stats:
                entry[f'S1{key}'] = winner_stats[key]
            for key in loser_stats:
                entry[f'S2{key}'] = loser_stats[key]

            entry['Outcome'] = 1
            entry['S1Elo'] = elo_model.get_elo(game['WTeamID'])
            entry['S2Elo'] = elo_model.get_elo(game['LTeamID'])
            entry['S1Seed'] = winner_seed
            entry['S2Seed'] = loser_seed
        else:
            # Loser gets slot 1
            for key in winner_stats:
                entry[f'S2{key}'] = winner_stats[key]
            for key in loser_stats:
                entry[f'S1{key}'] = loser_stats[key]

            entry['Outcome'] = 0
            entry['S2Elo'] = elo_model.get_elo(game['WTeamID'])
            entry['S1Elo'] = elo_model.get_elo(game['LTeamID'])
            entry['S2Seed'] = winner_seed
            entry['S1Seed'] = loser_seed

        entry['EloDiff'] = entry['S1Elo'] - entry['S2Elo']
        entry['SeedDiff'] = entry['S1Seed'] - entry['S2Seed']

        dataset.append(entry)

    elo_model.time_decay()

tournament_dataset = pd.DataFrame(dataset)

print('Dataset Generation Complete')

X = tournament_dataset.drop('Outcome', axis=1)
y = tournament_dataset['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Finally, we can use XGBoost
xgb_space = {
    'n_estimators': hp.uniformint('n_estimators', 100, 5000),
    'learning_rate': hp.uniform('learning_rate', 0.005, 0.2),
    'gamma': hp.uniform('gamma', 0, 10),
    'max_depth': hp.uniformint('max_depth', 3, 10),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.uniformint('min_child_weight', 1, 10),
}


def space_to_xgb(space):
    return XGBClassifier(subsample=0.8, seed=6, eval_metric=brier_score_loss, **space)


def xgb_objective(space):
    loss = []
    split = ShuffleSplit(n_splits=5, random_state=12)
    for train, test in split.split(X, y):
        model = space_to_xgb(space)
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        loss.append(brier_score_loss(y_test, y_pred[:,1]))

    return {'loss': sum(loss)/len(loss), 'status': hyperopt.STATUS_OK}


xgb_trials = Trials()
xgb_best_params = fmin(xgb_objective, xgb_space, algo=tpe.suggest, max_evals=100, trials=xgb_trials)
xgb_best_params['max_depth'] = xgb_best_params['max_depth'].astype(np.int64)
xgb_best_params['min_child_weight'] = xgb_best_params['min_child_weight'].astype(np.int64)
xgb_best_params['n_estimators'] = xgb_best_params['n_estimators'].astype(np.int64)
print(xgb_best_params)

xgb_model = space_to_xgb(xgb_best_params)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(brier_score_loss(y_test, y_pred))

y_pred = xgb_model.predict_proba(X_test)
print(brier_score_loss(y_test, y_pred[:,1]))

y_pred = []
for _, game in X_test.iterrows():
    rating1 = game['S1Elo']
    rating2 = game['S2Elo']

    y_pred.append(
         1 / (1 + 10 ** ((rating2 - rating1) / 400))
    )

print(brier_score_loss(y_test, y_pred))
print(brier_score_loss(y_test, np.array(y_pred) > 0.5))

stacked_model = StackingClassifier([
    ('elo', EloModelSklearn()),
    ('xgb', xgb_model)
])
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict_proba(X_test)
print(brier_score_loss(y_test, y_pred[:,1]))
