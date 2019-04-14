# DO NOT MODIFY THIS FILE.
#
# Any modifications to this file will not be used when grading your project.
# If you have any questions, please email the TAs.
#
# The main driver file for the project. You may edit this file to change which
# problems your Agent addresses while debugging and designing, but you should
# not depend on changes to this file for final execution of your project. Your
# project will be graded using our own version of this file.

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ProblemSet import ProblemSet
from Agent import Agent, SAME_IMAGE_MAX_RMS, two_means

def getNextLine(r):
    return r.readline().rstrip()

def plot_rms(name, rms_data, filename):
    sorted_data = np.sort(rms_data, axis=None)
    print(sorted_data)
    # print(np.trim_zeros(sorted_data))
    cluster = two_means(sorted_data)
    # print(cluster)
    cluster_range = [(np.min(cluster[i]), np.max(cluster[i])) for i in range(0, 2)]
    print(cluster_range)
    # plt.hist(self.rms_data, 10, facecolor='green', align='left', stacked=True)  # arguments are passed to np.histogram
    plt.gca().vlines(rms_data, 0, 1)
    # plt.axvline(self.rms_data)
    plt.axvspan(*cluster_range[0], facecolor='b', alpha=0.2)
    plt.axvspan(*cluster_range[1], facecolor='b', alpha=0.2)
    plt.axvline(SAME_IMAGE_MAX_RMS)
    # plt.axvline(cluster_range[1][0])
    plt.title(name + ' pairwise RMS')
    plt.savefig(filename)
    plt.clf()

def plot(dirname='pairwise_rms'):
    sets=[] # The variable 'sets' stores multiple problem sets.
            # Each problem set comes from a different folder in /Problems/
            # Additional sets of problems will be used when grading projects.
            # You may also write your own problems.

    r = open(os.path.join("Problems","ProblemSetList.txt"))    # ProblemSetList.txt lists the sets to solve.
    line = getNextLine(r)                                   # Sets will be solved in the order they appear in the file.
    while not line=="":                                     # You may modify ProblemSetList.txt for design and debugging.
        sets.append(ProblemSet(line))                       # We will use a fresh copy of all problem sets when grading.
        line=getNextLine(r)                                 # We will also use some problem sets not given in advance.

    agent=Agent()

    for set in sets:
        for problem in set.problems:   # Your agent will solve one problem at a time.
            #try:
            filename = os.path.join(dirname, problem.name.replace(' ', '-'))
            p = agent.get_problem(problem, filename)
            plot_rms(p.name, p.rms_data, filename)
    r.close()

# The main execution will have your agent generate answers for all the problems,
# then generate the grades for them.
def main(args):
    plot(args.dir)

if __name__ == "__main__":

    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("dir", help='Plot output directory')

    args = parser.parse_args()

    main(args)
