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
import math
# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops
import numpy as np

SAME_IMAGE_MAX_RMS = 0.15
SAME_DIFF_RMS_MAX = 0.001
SAME_INCREMENTAL_DIFF_STD = 0.0025

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
        # if problem.name != 'Basic Problem B-12':
            # return -1 
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
            sum += ret * 10

            ret = Agent.flip_vertically(dimension, matrix, choice)
            if ret > 0:
                self.log('flip_vertically', ret)
            sum += ret * 10

            ret = Agent.flip_diagnoally_top_left_to_bottom_right(dimension, matrix, choice)
            if ret > 0:
                self.log('flip_diagnoally_top_left_to_bottom_right', ret)
            sum += ret * 2

            ret = Agent.flip_diagnoally_top_right_to_bottom_left(dimension, matrix, choice)
            if ret > 0:
                self.log('flip_diagnoally_top_right_to_bottom_left', ret)
            sum += ret * 2

            ret = Agent.transpose_left_to_right(dimension, matrix, choice)
            if ret > 0:
                self.log('transpose_left_to_right', ret)
            sum += ret * 5

            ret = Agent.transpose_top_to_bottom(dimension, matrix, choice)
            if ret > 0:
                self.log('transpose_top_to_bottom', ret)
            sum += ret * 5

            ret = Agent.same_diff_vertically(dimension, matrix, choice)
            if ret > 0:
                self.log('same_diff_vertically', ret)
            sum += ret * 3

            ret = Agent.same_diff_horizontally(dimension, matrix, choice)
            if ret > 0:
                self.log('same_diff_horizontally', ret)
            sum += ret * 3

            ret = Agent.same_incremental_diff_vertically(dimension, matrix, choice)
            if ret > 0:
                self.log('same_incremental_diff_vertically', ret)
            sum += ret

            ret = Agent.same_incremental_diff_horizontally(dimension, matrix, choice)
            if ret > 0:
                self.log('same_incremental_diff_horizontally', ret)
            sum += ret

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
            '3x3': ['G*', 'AC', 'DF', 'BB', 'EE', 'HH']
        }

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        for pair in compare_pairs_map[dimension]:
            image0 = get_image(pair[0])
            image1 = get_image(pair[1])
            # print("flip_hor({})".format(pair), rms_diff(image0.transpose(Image.FLIP_LEFT_RIGHT), image1))
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
    def flip_diagnoally_top_left_to_bottom_right(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [ '**', 'AA', 'BC' ],
            # A B C
            # D E F
            # G H *
            '3x3': [ '**', 'AA', 'BD', 'CG', 'FH', 'EE']
        }
        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        for pair in compare_pairs_map[dimension]:
            image0 = get_image(pair[0])
            image1 = get_image(pair[1])
            if not is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270), image1):
                return 0

        return 1

    @staticmethod
    def flip_diagnoally_top_right_to_bottom_left(dimension, matrix, choice):
        compare_pairs_map = {
            # A B
            # C *
            '2x2': [ '*A', 'BB', 'CC' ],
            # A B C
            # D E F
            # G H *
            '3x3': [ '*A', 'BF', 'CC', 'DH', 'EE', 'GG']
        }
        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)
        for pair in compare_pairs_map[dimension]:
            image0 = get_image(pair[0])
            image1 = get_image(pair[1])
            # print(pair, rms_diff(image0.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90), image1))
            if not is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90), image1):
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
                if option is None:
                    if not is_same_image(image0, image1):
                        return 0
                else:
                    if not is_same_image(image0.transpose(option), image1):
                        return 0
            return 1

        sum = 0
        for option in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, None]:
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
                if option is None:
                    if not is_same_image(image0, image1):
                        return 0
                else:
                    if not is_same_image(image0.transpose(option), image1):
                        return 0

            return 1

        sum = 0
        for option in [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, None]:
            if compare_all(option):
                sum += 1

        return sum

    # For questions like Basic Problem B-10
    @staticmethod
    def same_diff_vertically(dimension, matrix, choice):
        if dimension == '3x3':
            return 0

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        # diff1 = ImageChops.difference(get_image('A'), get_image('C'))
        # diff2 = ImageChops.difference(get_image('B'), get_image('*'))

        diff1 = ImageChops.subtract(get_image('A'), get_image('C'), 2.0, 128)
        diff2 = ImageChops.subtract(get_image('B'), get_image('*'), 2.0, 128)

        if diff1 == diff2:
            return 1
        rms = rms_histogram(diff1, diff2)
        # print('horizontal', rms, choice.name)
        if rms < SAME_DIFF_RMS_MAX:
            return 1
        return 0

    @staticmethod
    def same_diff_horizontally(dimension, matrix, choice):
        if dimension == '3x3':
            return 0

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)
        # diff1 = ImageChops.difference(get_image('A'), get_image('C'))
        # diff2 = ImageChops.difference(get_image('B'), get_image('*'))

        diff1 = ImageChops.subtract(get_image('A'), get_image('B'), 2.0, 128)
        diff2 = ImageChops.subtract(get_image('C'), get_image('*'), 2.0, 128)
        if diff1 == diff2:
            return 1

        rms = rms_histogram(diff1, diff2)
        # print('verti', rms, choice.name)
        # print(diff1.histogram())
        # print(diff2.histogram())
        if rms < SAME_DIFF_RMS_MAX:
            return 1
        return 0

    @staticmethod
    def same_incremental_diff_vertically(dimension, matrix, choice):
        if dimension == '2x2':
            return 0
        columns = [['AD', 'DG'], ['BE', 'EH'], ['CF', 'F*']]

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        def get_diff(a1, b1, a2, b2):
            diff1 = ImageChops.difference(get_image(a1), get_image(b1))
            diff2 = ImageChops.difference(get_image(a2), get_image(b2))

            if diff1 == diff2:
                return 1
            rms = rms_diff(diff1, diff2)
            # print('rms({}-{},{}-{})'.format(a1, b1, a2, b2), rms)
            return rms 

        diffs = []
        for pairs in columns:
            diffs.append(get_diff(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]))
        # print('diffs', diffs)
        diff_of_diff = [diffs[2]-diffs[1], diffs[1] - diffs[0]]
        # print('diff_of_diff', diff_of_diff)

        # for m in range(0, 8):
        #     letter = chr(ord('A') + m)
        #     if is_same_image(get_image(letter), get_image('*')):
        #         return 0

        std = np.std(np.array(diff_of_diff))
        # print('std', std)
        if std < SAME_INCREMENTAL_DIFF_STD:
            return 1
        # if abs((diffs[2] - diffs[1]) - (diffs[1] - diffs[0])) < 0.1 * avg
        # print(diffs)
        # for i in range(len(diffs)):
        #     if i + 1 == len(diffs):
        #         break
        #     print('diffs[{}] - diffs[{}]'.format(i+1, i), diffs[i+1] - diffs[i])
        return 0

    @staticmethod
    def same_incremental_diff_horizontally(dimension, matrix, choice):
        if dimension == '2x2':
            return 0
        rows = [['AB', 'BC'], ['DE', 'EF'], ['GH', 'H*']]

        def get_image(name):
            if name == '*':
                return Image.open(choice.visualFilename)
            else:
                return Image.open(matrix.get(name).visualFilename)

        def get_diff(a1, b1, a2, b2):
            diff1 = ImageChops.difference(get_image(a1), get_image(b1))
            diff2 = ImageChops.difference(get_image(a2), get_image(b2))

            if diff1 == diff2:
                return 1
            rms = rms_diff(diff1, diff2)
            # print('rms({}-{},{}-{})'.format(a1, b1, a2, b2), rms)
            return rms
        
        diffs = []
        for pairs in rows:
            diffs.append(get_diff(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]))

        diff_of_diff = [diffs[2]-diffs[1], diffs[1] - diffs[0]]

        # for m in range(0, 8):
        #     letter = chr(ord('A') + m)
        #     if is_same_image(get_image(letter), get_image('*')):
        #         return 0

        std = np.std(np.array(diff_of_diff))
        if std < SAME_INCREMENTAL_DIFF_STD:
            return 1
        # if abs((diffs[2] - diffs[1]) - (diffs[1] - diffs[0])) < 0.1 * avg
        # print(diffs)
        # for i in range(len(diffs)):
        #     if i + 1 == len(diffs):
        #         break
        #     print('diffs[{}] - diffs[{}]'.format(i+1, i), diffs[i+1] - diffs[i])
        return 0

    # @staticmethod
    # def same_attribute_diff_horizontally(dimension, matrix, choice):
    #     if dimension == '3x3':
    #         return 0


    # @staticmethod
    # def same_attribute_diff_vertically(dimension, matrix, choice):
    #     if dimension == '3x3':
    #         return 0

def rms_histogram(image1, image2):
    # print(max(image1.histogram()))
    h1 = np.asarray(image1.histogram()) / float(max(image1.histogram()))
    # print('h1', h1)
    h2 = np.array(image2.histogram()) / float(max(image2.histogram()))
    # print('h2', h2)
    errors = h1 - h2
    return math.sqrt(np.mean(np.square(errors)))

def rms_diff(image1, image2):
    errors = np.asarray(ImageChops.difference(image1, image2)) / 255
    return math.sqrt(np.mean(np.square(errors)))
    # h = ImageChops.difference(image1, image2).histogram()
    # sum = np.array([value * ((i % 256) ** 2) for i, value in enumerate(h)]).sum()
    # return np.sqrt(sum / float(image1.size[0] * image1.size[1]))

def is_same_image(image1, image2):
    rms = rms_diff(image1, image2)
    # print('is_same_image', rms)
    return rms <= SAME_IMAGE_MAX_RMS