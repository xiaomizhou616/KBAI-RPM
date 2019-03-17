# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

from __future__ import print_function
import os

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops
#import numpy

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

        self.log('{}: {}'.format(problem.name, problem.problemType))

        # DEBUG
        # if problem.name != 'Basic Problem B-02':
        #     return -1 
        t = problem.problemType

        def get_subset(d, keys):
            ret = {}
            for k in keys:
                ret[k] = d.get(k)
            return ret
        if t == '2x2':
            matrix = get_subset(problem.figures, ['A', 'B', 'C'])
            choices = get_subset(problem.figures, [str(i) for i in range(1, 7)])
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
            self.log('trying choice {}'.format(c))
            fitness = self.calculate_fitness(t, matrix, choices[c])
            if fitness > max_fitness:
                answer = c
                max_fitness = fitness
        return int(answer)

    def calculate_fitness(self, t, m, c):
        sum = 0

        if Agent.all_identical(m, c):
            sum += 1
            self.log('Get one all_identical')

        return sum

    def log(self, *args):
        if self.debug_mode:
            print(*args)

    @staticmethod
    def all_identical(matrix, choice):

        # def all_equal(a, b):
        #     print('------------------')
        #     print('Compare the following:\n{}\n{}'.format(a, b))
        #     if hasattr(a, 'objects'):
        #         a_keys = list(a.objects.keys())
        #         b_keys = list(b.objects.keys())
        #         if len(a_keys) == 1 and len(a_keys) == len(b_keys):
        #             # One object in each. Ignore the name and compare attributes
        #             if not all_equal(a.objects.get(a_keys[0]), b.objects.get(b_keys[0])):
        #                 return False
        #         elif len(a_keys) == 2 and len(a_keys) == len(b_keys):
        #             # Two objects. Do two possible comparison
        #             compare1 = all_equal(a.objects.get(a_keys[0]), b.objects.get(b_keys[0])) and all_equal(a.objects.get(a_keys[1]), b.objects.get(b_keys[1]))
        #             compare2 = all_equal(a.objects.get(a_keys[0]), b.objects.get(b_keys[1])) and all_equal(a.objects.get(a_keys[1]), b.objects.get(b_keys[0]))
        #             if not compare1 and not compare2:
        #                 return False
        #         else:
        #             return False
        #     elif hasattr(a, 'attributes'):
        #         key_set = set(a.attributes.keys()) | set(b.attributes.keys())
        #         for k in key_set:
        #             if not all_equal(a.attributes.get(k), b.attributes.get(k)):
        #                 return False
        #     else:
        #         if a != b:
        #             return False
                
        #     return True

        def pixel_equal(a, b):
            image_a = Image.open(a.visualFilename)
            image_b = Image.open(b.visualFilename)
            return image_a == image_b

        for k in matrix.keys():
            if not pixel_equal(choice, matrix[k]):
                return False

        return True
