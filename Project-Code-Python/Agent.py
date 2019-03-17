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
import numpy as np

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
        # if problem.name != 'Basic Problem B-06':
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

        return self.find_fitted(t, matrix, choices, problem.hasVerbal, problem.hasVisual)

    def find_fitted(self, dimension, matrix, choices, has_verbal=False, has_visual=False):
        answer = "-1"
        max_fitness = .0
        for c in choices.keys():
            # calculate the fitness
            fitness = self.calculate_fitness(dimension, matrix, choices[c], has_verbal, has_visual)
            if fitness > max_fitness:
                answer = c
                max_fitness = fitness
        return int(answer)

    def calculate_fitness(self, dimension, matrix, choice, has_verbal, has_visual):
        sum = 0

        self.log('================= choice: {}'.format(choice.name))

        if has_visual:
            # See if every figure in matric pixel-equals to the choice
            ret = Agent.all_identical(matrix, choice)
            if ret > 0:
                self.log('all_identical', ret)
            sum += ret * 10

            ret = Agent.flip_horizontally(dimension, matrix, choice)
            if ret > 0:
                self.log('flip_horizontally', ret)
            sum += ret * 5

            ret = Agent.flip_vertically(dimension, matrix, choice)
            if ret > 0:
                self.log('flip_vertically', ret)
            sum += ret * 5

            ret = Agent.transpose_left_to_right(dimension, matrix, choice)
            if ret > 0:
                self.log('transpose_left_to_right', ret)
            sum += ret * 2

            ret = Agent.transpose_top_to_bottom(dimension, matrix, choice)
            if ret > 0:
                self.log('transpose_top_to_bottom', ret)
            sum += ret * 2

        return sum

    def log(self, *args):
        if self.debug_mode:
            print(*args)

    @staticmethod
    def all_identical(matrix, choice):

        def pixel_equal(a, b):
            image0 = Image.open(a.visualFilename)
            image1 = Image.open(b.visualFilename)
            return image0 == image1

        for k in matrix.keys():
            if not pixel_equal(choice, matrix[k]):
                return False

        return True

    @staticmethod
    def flip_horizontally(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [
                ['C', '*'],
                ['A', 'B']
            ],
            # A B C
            # D E F
            # G H *
            '3x3': [
                ['G', '*'],
                ['A', 'C'],
                ['D', 'F'],
                ['B', 'B'],
                ['E', 'E'],
                ['H', 'H']
            ]
        }

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        for pair in compare_pairs_map[dimension]:
            image0 = get_image(pair[0])
            image1 = get_image(pair[1])
            if not is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT), image1):
                return 0

        return 1

    @staticmethod
    def flip_vertically(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [
                ['B', '*'],
                ['A', 'C']
            ],
            # A B C
            # D E F
            # G H *
            '3x3': [
                ['C', '*'],
                ['A', 'G'],
                ['B', 'H'],
                ['D', 'D'],
                ['E', 'E'],
                ['F', 'F']
            ]
        }

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        for pair in compare_pairs_map[dimension]:
            image0 = get_image(pair[0])
            image1 = get_image(pair[1])
            if not is_same_image(image0.transpose(Image.FLIP_TOP_BOTTOM), image1):
                return 0

        return 1

    @staticmethod
    def transpose_left_to_right(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [ 'C*', 'AB' ],
            # A B C
            # D E F
            # G H *
            '3x3': [ 'H*', 'AB', 'BC', 'DE', 'EF', 'GH']
        }
        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)
        
        def compare_all(option):
            for pair in compare_pairs_map[dimension]:
                image0 = get_image(pair[0])
                image1 = get_image(pair[1])
                if not is_same_image(image0.transpose(option), image1):
                    return 0

            return 1

        sum = 0
        for option in [Image.ROTATE_90, Image.ROTATE_270]:
            if compare_all(option):
                sum += 1

        return sum

    @staticmethod
    def transpose_top_to_bottom(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [ 'B*', 'AC' ],
            # A B C
            # D E F
            # G H *
            '3x3': [ 'F*', 'AD', 'BE', 'CF', 'DG', 'EH']
        }
        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)
        
        def compare_all(option):
            for pair in compare_pairs_map[dimension]:
                image0 = get_image(pair[0])
                image1 = get_image(pair[1])
                # print('compare ', pair)
                if not is_same_image(image0.transpose(option), image1):
                    return 0

            return 1

        sum = 0
        for option in [Image.ROTATE_90, Image.ROTATE_270]:
            if compare_all(option):
                sum += 1

        return sum

def is_same_image(image1, image2):
    h = ImageChops.difference(image1, image2).histogram()
    sum = np.array([value * ((i % 256) ** 2) for i, value in enumerate(h)]).sum()
    rms = np.sqrt(sum / float(image1.size[0] * image1.size[1]))

    # print('is_same_image', rms)
    return rms <= 100