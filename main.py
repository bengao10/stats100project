"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a computational tool developed for STATS 100: Mathematics of Sports
to identify the merit-based rankings of the current top N ranked fighters.

The results can be used to inform match-ups, opposition strength, etc.

Author: Benjamen Gao
Initial & Last Edit: 5/4/2023 & 5/17/2023
Accompanying data is accessible https://github.com/bengao10/stats100project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# import relevant libraries
import numpy as np
import matplotlib.pyplot as plt

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
    equilibrium_vector, weight_class = calc_fighter_ranks(dict(), filename="Stats 100 MMA Data Sheet - LightweightOnlyTop15FINAL.csv")
    official_lightweight_rankings, official_welterweight_rankings = read_official_rankings()
    compare_official_and_predicted_rankings(official_lightweight_rankings, equilibrium_vector, weight_class)


# given rankings, official and predicted, creates a plot of the difference and returns the loss
def compare_official_and_predicted_rankings(official_rankings, equilibrium_vector, weight_class):
    # first fix container of predicted rankings
    predicted_rankings = sorted([(key, equilibrium_vector[key]) for key in equilibrium_vector], key=lambda x: x[1], reverse=True)
    predicted_rankings = [each_fighter[0] for each_fighter in predicted_rankings] # to get names

    official_rankings_map = dict()
    predicted_rankings_map = dict()

    # get in form {name: ranking}
    for i in range(len(official_rankings)):
        official_rankings_map[official_rankings[i]] = i

    for j in range(len(predicted_rankings)):
        predicted_rankings_map[predicted_rankings[j]] = j

    names = []
    total_error = 0
    all_official_points = []
    all_predicted_points = []
    # only consider fighters in the official rankings
    print(official_rankings_map)
    print(predicted_rankings_map)
    for fighter_key in official_rankings_map:

        # points are in form (official, predicted)
        names.append(fighter_key)
        print(fighter_key, official_rankings_map[fighter_key], predicted_rankings_map[fighter_key])
        all_official_points.append(official_rankings_map[fighter_key])
        all_predicted_points.append(predicted_rankings_map[fighter_key])
        total_error += abs(official_rankings_map[fighter_key] - predicted_rankings_map[fighter_key])

    # print(official_rankings_map)
    # print(predicted_rankings_map)
    # print(all_predicted_points)
    # print(all_official_points)

    plt.scatter(all_predicted_points, all_official_points)
    plt.title("Official vs Predicted Rankings")
    plt.xlabel("Predicted rankings")
    plt.ylabel("Official ranking")

    # plot names
    for i, name in enumerate(names):
        name = str(name[0] + ". " + name.split(" ")[-1])
        plt.text(all_predicted_points[i]+0.25, all_official_points[i]+0.25, name)

    plt.show()

    print(f"Total error for {weight_class} is {total_error}.")


# reads and stores in lists the official UFC rankings, as per the UFC rankings on May 4th 2023
def read_official_rankings():
    official_lightweight_rankings = []
    official_welterweight_rankings = []
    with open("Stats 100 MMA Data Sheet - OfficialLightweightRankings.csv") as official_lightweight_rankings_file:
        for line in official_lightweight_rankings_file:
            official_lightweight_rankings.append(line.strip())

    with open("Stats 100 MMA Data Sheet - OfficialWelterweightRankings.csv") as official_welterweight_rankings_file:
        for line in official_welterweight_rankings_file:
            official_welterweight_rankings.append(line.strip())

    return official_lightweight_rankings, official_welterweight_rankings


# prints the predicted rankings based on the equilibrium vector
def print_rankings(equilibrium_vector, weight_class):
    rankings = sorted([(key, equilibrium_vector[key]) for key in equilibrium_vector], key=lambda x: x[1], reverse=True)
    only_names = [each_fighter[0] for each_fighter in rankings]

    print(f"Predicted Rankings for UFC {weight_class} Division")
    print(f"~~~*****~~~   Champion: {only_names[0]}   ~~~*****~~~")
    for n in range(1, TOP_N):
        if n < 10:
            print(f"        # {n} contender : {only_names[n]}")
        else:
            print(f"        #{n} contender : {only_names[n]}")


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
def read_csv(filename):
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

    return "Lightweight" if "light" in filename.lower() else "Welterweight"


# encode equilibrium rankings into a dictionary for readability
def calc_fighter_ranks(fighter_rank, filename="UFCLightweightsTest.csv"):
    weight_class = read_csv(filename)
    equilibrium_matrix = update_ranking(calc_equilibrium_matrix(make_matrix()))
    for each_fighter in equilibrium_matrix:
        fighter_rank[each_fighter.name] = each_fighter.equilibriumVal
    return fighter_rank, weight_class


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~ Code Proper Ends Here ~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


if __name__ == "__main__":
    main()
