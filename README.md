# March Madness Predictions

In this project, I seek to predict the results of the NCAA Men's and Women's D1 Basketball Tournament. I aim 
achieve this goal through using a variety of techniques. I will attempt to use an elo system as a baseline
for the rest of the models, and then apply XGBoost to data involving averages from the entire season
to generate more holistic and data-driven predictions than just the elo. I also want to look into the 
predictive power of recent games, using a neural network to try to see if it can find a trend in recent
stats with win-shares. 

## Project Requirements
This project uses pandas, numpy, and hyperopt. Verified to work using Python 3.10.9 on a 2021 M1 MacBook
Air. Use either command to install the needed packages:
```bash
conda install pandas numpy hyperopt
```
```bash
pip install pandas numpy hyperopt
```
Data comes from Kaggle's March Machine Learning Mania Competition. Get the data by going to
[kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data), agreeing to the
competition rules, then downloading the dataset. Put this data in the ```./data``` folder. 

## The Elo Model
The most important factor of an elo model is the K-factor, which is modified throughout the model in a few
ways. First, there is a change to it based off of margin of victory.
[FiveThirtyEight](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) gives some
insight into the process. Their approach is to take the margin of victory and pass it through some
functions that result in a multiplier to the K-factor of the ELO ranking. This factor takes into account
the difference in ELO to make the result much more accurate. Another modification from the algorithm takes 
insight from [Harvard Sports Analysis](https://harvardsportsanalysis.org/2019/01/a-simple-improvement-to-fivethirtyeights-nba-elo-model/)
where they apply a time-sensitive dampening to the K-factor of the system. This results in earlier games
having bigger swings to the ELO because of changes that can happen off-court during the off-season. In this
project, exponential decay is used to model this dampening. In the off-season, each team also loses ELO and 
reverts to the mean.