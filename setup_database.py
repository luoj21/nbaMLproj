import mysql
import mysql.connector


class MySQLConnector():
    """Connects user to MySQL database"""
    
    def __init__(self, host: str, user: str, password: str):
        self.host = host
        self.user = user
        self.password = password

    def connect_to_db(self):
        """Connect user to the database"""
        self.mydb = mysql.connector.connect(
        host= self.host,
        user= self.user,
        password = self.password
      )
        
        print("## Connected to database ##")
        
        
    def disconnect_from_db(self):
        """Disconnect from database"""
        self.mydb.cursor().close()
        self.mydb.close()
        del self.mydb
        print("## Disconnected from database ##")
      
        
    def create_nba_db(self):
        """Create nba database"""
        self.mydb.cursor().execute("CREATE DATABASE IF NOT EXISTS NBA_DB")
        
    def create_stats_table(self):
      """Create table for player statistics"""  
      self.mydb.cursor().execute("USE NBA_DB")
      self.mydb.cursor().execute("""
                      CREATE TABLE IF NOT EXISTS PER_GAME_STATS (
                      player_id INT NOT NULL,
                      season INT NOT NULL,       
                      pos VARCHAR(50),
                      age INT,
                      team VARCHAR(50),
                      games FLOAT,
                      games_started FLOAT,  
                      mp FLOAT,
                      fg FLOAT,
                      fga FLOAT,
                      fg_perc FLOAT,
                      three_point FLOAT,
                      three_point_att FLOAT,
                      three_point_perc FLOAT,
                      two_point FLOAT,
                      two_point_att FLOAT,
                      two_point_perc FLOAT,
                      efg FLOAT,     
                      ft FLOAT,
                      fta FLOAT,
                      ft_perc FLOAT,
                      orb FLOAT,
                      drb FLOAT,
                      trb FLOAT,
                      ast FLOAT,
                      stl FLOAT,
                      blk FLOAT,
                      tov FLOAT,
                      pf FLOAT,
                      pts FLOAT,
                      PRIMARY KEY (player_id, season),
                      FOREIGN KEY (player_id) REFERENCES PER_SEASON_INCOME(player_id));
      """)

    
    def load_data(self, file_path: str, table_name: str):
      """Load data to specified table"""
      self.mydb.cursor().execute("USE NBA_DB")
      self.mydb.cursor().execute(f"""LOAD DATA INFILE '{file_path}'
                      INTO TABLE {table_name}
                      FIELDS TERMINATED BY ','
                      LINES TERMINATED BY '\n' 
                      IGNORE 1 LINES;""")

      self.mydb.commit()


    def create_income_table(self):
        """Create income table"""
        self.mydb.cursor().execute("USE NBA_DB")
        self.mydb.cursor().execute("""CREATE TABLE IF NOT EXISTS PER_SEASON_INCOME (
                                   player_id INT NOT NULL,
                                   season INT NOT NULL,
                                   income FLOAT,
                                   adj_income FLOAT,
                                   PRIMARY KEY (player_id, season),
                                   FOREIGN KEY (player_id) REFERENCES PLAYERS_TABLE(player_id));""")


    def create_players_table(self):
      """Create players table"""
      self.mydb.cursor().execute("USE NBA_DB")
      self.mydb.cursor().execute("""CREATE TABLE IF NOT EXISTS PLAYERS_TABLE (
                                 player_id INT NOT NULL,
                                 player VARCHAR(100),
                                 PRIMARY KEY (player_id)
      );""")
      self.mydb.commit()
       

    def clear_table(self, database: str, table: str):
        """Clear specified table"""
        self.mydb.cursor().execute(f"USE {database}")
        self.mydb.cursor().execute(f"""TRUNCATE TABLE {table}""")



"""Original DB structure

t1: Income
(Season , player_id), player, income

t2: Stats
(Season, player_id), player, ppg, stls, reb, etc...

Normalized DB structure

t1: Income
(Season (CPK/FK), player_id, (CPK/FK)) income

t2: Stats
(Season (CPK/FK), player_id, (CPK/FK)), ppg, stls, reb, etc...

t3: Player Names
player_id (PK), player
"""


if __name__ == "__main__":
  
  data_path = '/Users/jasonluo/Documents/nbaProj/data'
  mysql_connector = MySQLConnector(host="***", user="***", password="***")
  mysql_connector.connect_to_db()
  mysql_connector.create_nba_db()
  mysql_connector.create_players_table()
  mysql_connector.create_income_table()
  mysql_connector.create_stats_table()

  # Uploading data to database
  mysql_connector.load_data(f'{data_path}/transformed_data/all_players_data.csv', 'PLAYERS_TABLE')
  mysql_connector.load_data(f'{data_path}/transformed_data/all_season_income.csv', 'PER_SEASON_INCOME')
  mysql_connector.load_data(f'{data_path}/transformed_data/all_player_stats.csv', 'PER_GAME_STATS')