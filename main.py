import pandas as pd
import numpy as np
import hyperopt

from hyperopt import hp, fmin, Trials, tpe


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

    for year in range(2004, 2013):
        results = compact_results_rs[compact_results_rs['Season'] == year]
        for _, game in results.iterrows():
            wid, lid = game['WTeamID'], game['LTeamID']
            mov = game['WScore'] - game['LScore']

            home_court = None
            if game['WLoc'] == 'H':
                home_court = 1
            elif game['WLoc'] == 'A':
                home_court = 0

            elo_model.update_rankings(wid, lid, result=1, home_team=home_court,
                                      margin_of_victory=mov, day_num=game['DayNum'])

        tourny_results = compact_results_mm[compact_results_mm['Season'] == year]
        for _, game in tourny_results.iterrows():
            wid, lid = game['WTeamID'], game['LTeamID']
            mov = game['WScore'] - game['LScore']

            elo_model.update_rankings(wid, lid, result=1, home_team=None,
                                      margin_of_victory=mov, day_num=game['DayNum'])

        elo_model.time_decay()

    loss = 0
    games = 0

    for year in range(2013, 2023):
        # Same, but everything counts for Brier Score, and no training in the tournament

        results = compact_results_rs[compact_results_rs['Season'] == year]
        for _, game in results.iterrows():
            wid, lid = game['WTeamID'], game['LTeamID']
            mov = game['WScore'] - game['LScore']

            home_court = None
            if game['WLoc'] == 'H':
                home_court = 1
            elif game['WLoc'] == 'A':
                home_court = 0

            prediction = elo_model.predict_game(wid, lid, home_team=home_court)
            # loss += (1 - prediction) ** 2
            # games += 1

            elo_model.update_rankings(wid, lid, result=1, home_team=home_court,
                                      margin_of_victory=mov, day_num=game['DayNum'])

        tourny_results = compact_results_mm[compact_results_mm['Season'] == year]
        for _, game in tourny_results.iterrows():
            wid, lid = game['WTeamID'], game['LTeamID']

            prediction = elo_model.predict_game(wid, lid, home_team=None)
            loss += (1 - prediction) ** 2
            games += 1

        elo_model.time_decay()

    return {'loss': loss / games, 'status': hyperopt.STATUS_OK}

elo_trials = Trials()
elo_best_params = fmin(elo_objective, elo_space, algo=tpe.suggest, max_evals=100, trials=elo_trials)

print(elo_best_params)
