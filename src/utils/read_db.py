import sqlite3
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
db_path = os.path.join(base_path, 'data', 'lung_cancer.db')


def get_df():
    table_name = "lung_cancer"

    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * from {}".format(table_name), con)
    con.close()
    return df


def check_path():
    return db_path
