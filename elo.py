# ELO Ranking Algorithm for teams

import math

class ELOModel:
    def __init__(self, *, starting_rating, teams, end_of_season_decay,
                 K, K_multiplier, K_decay, home_court_advantage,
                 margin_of_victor_multiplier):
        self.ratings = {team: starting_rating for team in teams}
        self.end_of_season_decay = end_of_season_decay
        self.K_multiplier = K_multiplier
        self.K_decay = K_decay
        self.mov_mult = margin_of_victor_multiplier
        self.K = K
        self.hca = home_court_advantage

    def predict_game(self, team1, team2, home_team=None):
        # Expected win-share of team_1 playing team_1

        rating1 = self.ratings[team1]
        rating2 = self.ratings[team2]

        if home_team is not None and home_team in (0, 1):
            rating1 += home_team * self.hca
            rating2 += (1 - home_team) * self.hca

        # Formula is 1/(1 + 10^[(r2-r1)/400]
        # It's a logistic curve

        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def update_rankings(self, team1, team2, result, home_team=None, margin_of_victory=None, day_num=None):
        # 1 is team1 wins, 0 is team1 loses
        expected = self.predict_game(team1, team2)

        rating1 = self.ratings[team1]
        rating2 = self.ratings[team2]

        if home_team is not None and home_team in (0, 1):
            rating1 += home_team * self.hca
            rating2 += (1 - home_team) * self.hca

        K_mult = 1
        if margin_of_victory is not None:
            winner_diff = rating1 - rating2 if result == 1 else rating2 - rating1
            K_mult *= (margin_of_victory + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)

        if day_num is not None:
            K_mult *= self.K_multiplier * math.exp(-day_num * self.K_decay / 10)

        # K-factor * (result - expected)
        delta = self.K * K_mult * (result - expected)

        self.ratings[team1] += delta
        self.ratings[team2] -= delta

    def time_decay(self):
        mean = sum(x for x in self.ratings.values()) / len(self.ratings)
        for team in self.ratings:
            rating = self.ratings[team]
            delta = (mean - rating) * self.end_of_season_decay
            self.ratings[team] = rating + delta
