"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a computational tool developed for STATS 100: Mathematics of Sports
to identify the merit-based rankings of the current top N ranked fighters.

The results can be used to inform match-ups, opposition strength, etc.

Author: Benjamen Gao
Initial & Last Edit: 5/4/2023 & 5/14/2023
Accompanying data is accessible https://github.com/bengao10/stats100project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# import relevant libraries
import numpy as np

# fighter is a class to represent fighter objects
from fighter import fighter

# key global variables
TOP_N = 16  # consider top n players
POWER = 20  # raise Markov to the POWER
ALPHA = 0.85  # standard noise for PageRank
fighters = []  # list of all recorded fighters
sum_columns = [0] * TOP_N


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ Code Proper Begins Below ~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# general master function to operate code
def main():
    equilibrium_vector = calc_fighter_ranks(dict())
    for key in equilibrium_vector:
        for key2 in equilibrium_vector:
            print(f"{key} vs {key2}: {equilibrium_vector[key] / equilibrium_vector[key2]}")


# returns fighters in sorted order based on calculated rankings
def update_ranking(equilibrium_matrix: np.matrix):
    for i in range(TOP_N):
        fighters[i].equilibriumVal = equilibrium_matrix[i, 1]
    fighters.sort(reverse=True)
    return fighters


# uses PageRank Algorithm on input fighter data matrix to get rankings
# see here for additional information https://en.wikipedia.org/wiki/PageRank
def calc_equilibrium_matrix(sparse_matrix: np.array):
    w_prime = ALPHA * sparse_matrix + (1 - ALPHA) * (np.ones((TOP_N, TOP_N)))
    sum_of_cols = [0 for _ in range(16)]

    # sum of the columns
    for i in range(len(w_prime)):
        for j in range(len(w_prime[0])):
            sum_of_cols[j] += w_prime[i, j]

    # normalize each entry with respect to the column
    for i in range(len(w_prime)):
        for j in range(len(w_prime[0])):
            w_prime[i, j] /= sum_of_cols[j]

    # calculate and return equilibrium matrix (by Perron-Frobenius theorem)
    equilibrium_matrix = np.linalg.matrix_power(w_prime, POWER)
    return equilibrium_matrix


# create a matrix to encode the provided fighter data
def make_matrix():
    matrix = np.array([each_fighter.record for each_fighter in fighters])

    # normalize the matrix
    for i in range(TOP_N):
        for j in range(TOP_N):
            if sum_columns[j] != 0:
                matrix[i, j] /= sum_columns[j]
            else:
                matrix[i, j] = 1 / TOP_N
    return matrix


# read csv with provided fighter data
def read_csv(filename="UFCLightWeightsTest.csv"):
    global sum_columns
    sum_columns = [0] * TOP_N
    with open(filename) as csv:

        # pre-set global variable fighters
        global fighters
        fighters = []
        data = csv.readlines()

        # extract fighter data from csv
        for i in range(1, len(data)):
            line = data[i].strip()
            information = line.split(",")
            record = [float(diff) for diff in information[1:]]

            for j in range(len(record)):
                sum_columns[j] += record[j]
            fighters.append(fighter(information[0], record))
        # start at idx 1 since idx 0 is the fighter name


# encode equilibrium rankings into a dictionary for readability
def calc_fighter_ranks(fighter_rank):
    read_csv()
    equilibrium_matrix = update_ranking(calc_equilibrium_matrix(make_matrix()))
    for each_fighter in equilibrium_matrix:
        fighter_rank[each_fighter.name] = each_fighter.equilibriumVal
    return fighter_rank


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~ Code Proper Ends Here ~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


if __name__ == "__main__":
    main()
