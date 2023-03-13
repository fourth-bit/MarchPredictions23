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

