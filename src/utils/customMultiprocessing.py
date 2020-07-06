import multiprocessing
from itertools import repeat
import numpy as np

def customMultiprocessing(f, params_list, pool_size=8):
    '''
    Execute a function f using multiprocessing \n
    :param f: function to exectute
    :param params_list: list of list of params. Element inside must be of the same lenght (number of iterations) or of lenght 1. In case of lenght 1 they will be repeated. MUST be a list of lists!
    :param pool_size: number of processes to use
    :return: list of element containing the return of each execution of the function
    '''
    print("Inizio multiprocessing")
    lengths = [len(el) if type(el) == list else 1 for el in params_list]
    base_el = np.max(lengths)
    for lens in lengths:
        if (lens == base_el) or (lens == 1):
            continue
        else:
            raise Exception(
                "Params list must have elements of the same lengths or of length 1, found: {}".format(lengths))
    final_params = [el if type(el) == list else [el]*base_el for el in params_list]
    print("parameters ok")
    pool = multiprocessing.Pool(pool_size)
    print("pool fatto")
    return pool.starmap(f, zip(*final_params))


# faccio una funzione che dato un nome ti ritorna un cognome random
def surnameFunction(name, index):
    return name, str(index) + "surnameeee"


if __name__ == "__main__":
    possible_names = ["noemi", "federico", "isola d'elba"]
    result = customMultiprocessing(surnameFunction, [possible_names, 1])
    for el in result:
        print(el)
    result_dict = {}
