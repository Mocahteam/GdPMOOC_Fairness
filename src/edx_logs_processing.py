import json
import pandas as pd
import glob
from typing import List, Callable
from time import time
from tqdm import tqdm


def read_log_file(
    fichier: str, verbose: bool = False, show_error: bool = False
) -> pd.DataFrame:
    """Read an OpenEdX log file and return a dataframe

    Args:
        fichier (str): filename with its path
        verbose (bool, optional): to print additional info. Defaults to False.
        show_error (bool, optional): to print the error message. Defaults to False.

    Returns:
        pd.DataFrame: the dataframe with the log file content
    """
    with open(fichier, "r") as f:
        string = f.read()

    # transformation en tableau des index
    content = string.splitlines()

    logs = []
    f = json.loads(content[0])
    for k in range(len(content)):
        try:
            f = json.loads(content[k])
            logs.append(f)
        except:
            if show_error:
                print("probleme sur le fichier", fichier, "à la ligne", k, "\n")
    df = pd.DataFrame(logs, columns=logs[0].keys())
    n = len(df)
    df = df[df["username"] != ""]  # enleve les username vide
    df = df.reset_index(drop=True)
    if verbose:
        print(
            n - len(df),
            "lignes sans username supprimées, soit :",
            (n - len(df)) / n * 100,
            "%",
        )
    return df


def read_all_log_files(
    path: str,
    mask: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    verbose=False,
    time_it: bool = True,
) -> pd.DataFrame:
    """Read all log files in a folder and return a dataframe - possibility to call a function on files as they are read

    Args:
        path (str): path to the folder
        mask (_type_, optional): raw pretreatment function to be applied on each file read. Defaults to lambdax:x.
        verbose (bool, optional): print additional info when processing. Defaults to False.
        TIME (bool, optional): show the time needed to compute. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """

    liste_fichiers: List[str] = glob.glob(path + "*.log", recursive=True)
    liste_fichiers.sort()
    t = time()
    df_list = [mask(read_log_file(f, verbose=verbose)) for f in tqdm(liste_fichiers)]
    if time_it:
        print("Import time:", time() - t)
    return pd.concat(df_list, ignore_index=True)


def filter_events(df: pd.DataFrame, column: str, event_list: List[str]) -> pd.DataFrame:
    """filtre qui ne garde que les lignes dont la colonne 'colonne' est dans la liste_filtre"""
    res = df[df[column].isin(event_list)]
    return res.reset_index(drop=True)


def filter_users(df: pd.DataFrame, column: str, user_list: List[str]) -> pd.DataFrame:
    """filtre qui ne garde que les lignes dont la colonne 'colonne' contenant des username ou userid est dans la user_list"""
    res = df[df[column].isin(user_list)]
    return res.reset_index(drop=True)
    

def filter_dates(df: pd.DataFrame, column: str, start_date: str, end_date: str) -> pd.DataFrame:
    """filtre qui ne garde que les lignes dont la colonne 'colonne' contenant des dates est inclus dans l'intervale de date [start_date, end_date]"""
    res = df[ (df[column] >= start_date) & (df[column] <= end_date) ]
    return res.reset_index(drop=True)