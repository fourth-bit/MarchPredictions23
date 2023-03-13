import random

import hyperopt
import numpy as np
import pandas as pd

from hyperopt import hp, fmin, tpe, Trials
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import ShuffleSplit

from elo import *


class Model:
    def __init__(self, rs_compact, mm_compact, rs_detailed, mm_detailed, mm_seeds):
        self.compact_results_rs = pd.read_csv(rs_compact)
        self.compact_results_mm = pd.read_csv(mm_compact)
        self.detailed_results_rs = pd.read_csv(rs_detailed)
        self.detailed_results_mm = pd.read_csv(mm_detailed)
        self.mm_seeds = pd.read_csv(mm_seeds)
        self.mm_seeds['Seed'] = self.mm_seeds['Seed'].apply(lambda x: float(x[1:3]))

        # Get all team IDs
        self.teams = np.unique(
            self.compact_results_rs[['WTeamID', 'LTeamID']].to_numpy().flatten()
        )

        self.elo_best_params = None

        self.dataset = None
        self.X = None
        self.y = None

        self.xgb_best_params = None
        self.xgb_model = None
        self.stacked_model = None

    def space_to_elo(self, space):
        return ELOModel(teams=self.teams, starting_rating=1200, **space)

    def fit_elo(self, *, max_evals=100):
        elo_space = {
            'end_of_season_decay': hp.uniform('end_of_season_decay', 0, 1),
            'K': hp.uniform('K', 10, 50),
            'K_multiplier': hp.uniform('K_multiplier', 1, 4),
            'K_decay': hp.uniform('K_decay', 0, 1),
            'home_court_advantage': hp.uniform('home_court_advantage', 30, 200),
            'margin_of_victor_multiplier': hp.uniform('margin_of_victor_multiplier', 0, 1),
        }

        elo_trials = Trials()
        self.elo_best_params = fmin(self.elo_objective, elo_space, algo=tpe.suggest, max_evals=max_evals, trials=elo_trials)
        return elo_trials

    def elo_objective(self, space):
        elo_model = self.space_to_elo(space)

        for year in range(1993, 2013):
            results = self.compact_results_rs[self.compact_results_rs['Season'] == year]
            simulate_games(elo_model, results)

            elo_model.time_decay()

        loss = 0
        count = 0

        for year in range(2013, 2023):
            # Same, but tourney counts towards predictive score

            results = self.compact_results_rs[self.compact_results_rs['Season'] == year]
            simulate_games(elo_model, results)

            tourny_results = self.compact_results_mm[self.compact_results_mm['Season'] == year]
            for _, game in tourny_results.iterrows():
                wid, lid = game['WTeamID'], game['LTeamID']

                prediction = elo_model.predict_game(wid, lid, home_team=None)
                loss += (1 - prediction) ** 2
                count += 1

            elo_model.time_decay()

        return {'loss': loss / count, 'status': hyperopt.STATUS_OK}

    def build_dataset(self):
        if self.elo_best_params is None:
            raise ValueError('Must run fit_elo before running build_dataset. The elo model is required information for it')

        seasonal_averages = {}
        seasonal_weighted_averages = {}

        for season in range(1985, 2024):
            games = self.detailed_results_rs[self.detailed_results_rs['Season'] == season]
            seasonal_averages[season] = {}
            seasonal_weighted_averages[season] = {}

            for team in self.teams:
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
        elo_model = self.space_to_elo(self.elo_best_params)
        dataset = []

        # Simulate until 2003 using same data
        for season in range(1993, 2003):
            regular_season_games = self.compact_results_rs[self.compact_results_rs['Season'] == season]
            simulate_games(elo_model, regular_season_games)

        # Now actually build the dataset
        for season in range(2003, 2024):
            regular_season_games = self.compact_results_rs[self.compact_results_rs['Season'] == season]
            simulate_games(elo_model, regular_season_games)

            tournament_games = self.compact_results_mm[self.compact_results_mm['Season'] == season]
            tournament_seeds = self.mm_seeds[self.mm_seeds['Season'] == season]

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

        self.dataset = pd.DataFrame(dataset)
        self.X = self.dataset.drop('Outcome', axis=1)
        self.y = self.dataset['Outcome']


    def space_to_xgb(self, space):
        return XGBClassifier(subsample=0.8, seed=6, eval_metric=brier_score_loss, **space)

    def xgb_objective(self, space):
        loss = self.cross_val_score(lambda: self.space_to_xgb(space))
        return {'loss': sum(loss)/len(loss), 'status': hyperopt.STATUS_OK}

    def cross_val_score(self, model_gen):
        loss = []
        split = ShuffleSplit(n_splits=5, random_state=12)
        for train, test in split.split(self.X, self.y):
            model = model_gen()
            X_train, y_train = self.X.iloc[train], self.y.iloc[train]
            X_test, y_test = self.X.iloc[test], self.y.iloc[test]
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            loss.append(brier_score_loss(y_test, y_pred[:, 1]))
        return loss

    def train_models(self):
        if self.X is None or self.y is None:
            raise ValueError('Must run build_dataset before train_models')

        # Finally, we can use XGBoost
        xgb_space = {
            'n_estimators': hp.uniformint('n_estimators', 100, 5000),
            'learning_rate': hp.uniform('learning_rate', 0.005, 0.2),
            'gamma': hp.uniform('gamma', 0, 10),
            'max_depth': hp.uniformint('max_depth', 3, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.uniformint('min_child_weight', 1, 10),
        }

        xgb_trials = Trials()
        xgb_best_params = fmin(self.xgb_objective, xgb_space, algo=tpe.suggest, max_evals=100, trials=xgb_trials)
        xgb_best_params['max_depth'] = xgb_best_params['max_depth'].astype(np.int64)
        xgb_best_params['min_child_weight'] = xgb_best_params['min_child_weight'].astype(np.int64)
        xgb_best_params['n_estimators'] = xgb_best_params['n_estimators'].astype(np.int64)
        self.xgb_best_params = xgb_best_params

        self.xgb_model = self.space_to_xgb(self.xgb_best_params)
        self.xgb_model.fit(self.X, self.y)

        stacked_model_gen = lambda: StackingClassifier([('elo', EloModelSklearn()), ('xgb', self.xgb_model)])
        loss = self.cross_val_score(stacked_model_gen)
        print(loss, sum(loss)/len(loss))

        self.stacked_model = stacked_model_gen()
        self.stacked_model.fit(self.X, self.y)
