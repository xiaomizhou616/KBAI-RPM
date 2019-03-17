# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
#from PIL import Image
#import numpy

from __future__ import print_function 
import os

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self.debug_mode = (os.getenv('XH_DEBUG', "").lower() in ['1', 'true', 'on'])

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):

        self.log(problem.problemType)
        t = problem.problemType

        def get_subset(d, keys):
            ret = {}
            for k in keys:
                ret[k] = d.get(k)
            return ret
        if t == '2x2':
            matrix = get_subset(problem.figures, ['A', 'B', 'C'])
            choices = get_subset(problem.figures, [str(i) for i in range(1, 4)])
        elif t == '3x3':
            matrix = get_subset(problem.figures, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
            choices = get_subset(problem.figures, [str(i) for i in range(1, 9)])
        else:
            self.log("Unknown problem type {}".format(t))
            return -1

        if not problem.hasVerbal:
            self.log("verbal description is needed")
            return -1

        return self.find_fitted(t, matrix, choices)

    def find_fitted(self, t, matrix, choices):
        answer = "-1"
        max_fitness = .0
        for c in choices.keys():
            # calculate the fitness
            fitness = self.calculate_fitness(t, matrix, choices[c])
            if fitness > max_fitness:
                answer = c
                max_fitness = fitness
        return int(answer)

    def calculate_fitness(self, t, m, c):
        sum = 1

        return sum

    def log(self, *args):
        if self.debug_mode:
            print(*args)
