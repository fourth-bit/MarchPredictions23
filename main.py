from model import *


mens_model = Model(
    rs_compact='data/MRegularSeasonCompactResults.csv',
    mm_compact='data/MNCAATourneyCompactResults.csv',
    rs_detailed='data/MRegularSeasonDetailedResults.csv',
    mm_detailed='data/MNCAATourneyDetailedResults.csv',
    mm_seeds='data/MNCAATourneySeeds.csv',
)

mens_model.fit_elo()
mens_model.build_dataset()
mens_model.train_models()
mens_predictions = mens_model.predict2023()
mens_predictions_elo = mens_model.predict2023(only_elo=True)

print('Starting Women\'s')
womens_model = Model(
    rs_compact='data/WRegularSeasonCompactResults.csv',
    mm_compact='data/WNCAATourneyCompactResults.csv',
    rs_detailed='data/WRegularSeasonDetailedResults.csv',
    mm_detailed='data/WNCAATourneyDetailedResults.csv',
    mm_seeds='data/WNCAATourneySeeds.csv',
)

womens_model.fit_elo()
womens_model.build_dataset()
womens_model.train_models()
womens_predictions = womens_model.predict2023()
womens_predictions_elo = womens_model.predict2023(only_elo=True)

predictions = pd.concat([mens_predictions, womens_predictions])
predictions.to_csv('submission.csv')
elo_predictions = pd.concat([mens_predictions_elo, womens_predictions_elo])
elo_predictions.to_csv('submission_elo.csv')
