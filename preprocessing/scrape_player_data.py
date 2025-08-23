import numpy as np 
import pandas as pd
import requests
import time
import collections
from bs4 import BeautifulSoup
from io import StringIO


def request_stats_data(season1: int, season2: int):
    """Obtains HTML from NBA Reference for the seasons specified.
    The seasons are denoted by the year that they end. So '2024' corresponds to the 2023-2024 season"""

    years = list(range(season1, season2 + 1))
    for year in years:
        url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
        data = requests.get(url)
        time.sleep(15)

        assert data.status_code < 400
        
        with open(f"data/stats_html/{year}.html", "w+") as f:
            f.write(data.text)



def request_income_data(season1: int, season2: int):
    """Obtains HTML from HoopsHype for the seasons specified.
    The seasons are denoted by the year that they end. So '2024' corresponds to the 2023-2024 season"""
    years = list(range(season1, season2 + 1))
    for year in years:
        url = f'https://hoopshype.com/salaries/players/{year - 1}-{year}/'
        data = requests.get(url)
        time.sleep(10)

        assert data.status_code < 400
        
        with open(f"data/income_html/{year}.html", "w+") as f:
            f.write(data.text)




def export_stats_data(season: int):
    """Exports a CSV for a particular season that contains per game stats
    for each player"""
    with open(f'data/stats_html/{season}.html') as f:
        page = f.read()

    soup = BeautifulSoup(page, 'html.parser')
    season_data = soup.find('table' ,id = "per_game_stats")
    season_data_df = pd.concat(pd.read_html(StringIO(str(season_data))))
    season_data_df['season'] = season
    season_data_df = season_data_df[season_data_df['Rk'] != 'Rk'] # extra layer of filtering
    season_data_df.drop(columns = 'Rk', inplace = True)

    #season_data_df.to_csv(f"/Users/jasonluo/Documents/nbaProj/per_game_stats_data/{season}_player_data.csv", 
                          #index = False, na_rep=r'\N')
    
    season_data_df.to_csv(f"/Users/jasonluo/Documents/nbaProj/data/per_game_stats_data/{season}_player_data.csv", 
                          index = False, na_rep=np.NaN)
    


def export_salary_data(season: int):
    """Exports a CSV for a particular season that contains annual income 
    for each player that played"""
    with open(f'data/income_html/{season}.html') as f:
        page = f.read()

    soup = BeautifulSoup(page, 'html.parser')
    incomes = soup.find('div', class_ = 'hh-salaries-ranking')

    incomes_df = pd.concat(pd.read_html(StringIO(str(incomes))))
    incomes_df.drop(incomes_df.columns[0], axis = 1, inplace = True)
    incomes_df.rename(columns={incomes_df.columns[1]: "income", incomes_df.columns[2]: "adj_income"}, inplace = True)

    for index, row in incomes_df.iterrows():
        incomes_df.loc[index, 'income'] = int(incomes_df['income'][index][1:].replace(',', ''))
        incomes_df.loc[index, 'adj_income'] = int(incomes_df['adj_income'][index][1:].replace(',', ''))

    
    incomes_df['season'] = season
    incomes_df = incomes_df.sort_values('Player', ascending=True)


    incomes_df.to_csv(f"/Users/jasonluo/Documents/nbaProj/data/player_season_income/{season}_income_data.csv", 
                          index = False, na_rep=np.NaN)


def check_player_duplicates(df: pd.DataFrame):
    """Checks if duplicate players are in data"""
    if len(df['Player']) != len(pd.unique(df['Player'])):
        print([item for item, count in collections.Counter(df["Player"]).items() if count > 1])

