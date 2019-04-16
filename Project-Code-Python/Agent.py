from __future__ import print_function
import os
import math
# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops
import numpy as np
# import matplotlib.pyplot as plt

# np.set_printoptions(linewidth=200)

# Select a specific question to evaluate, will be evaluated as name.startswith(PROBLEM_STARTS_WITH)
# - '' (empty string) means all
# - 'Basic Problem' means all basic problems
# - 'Basic Problem B-10' means only basic problem B-10
PROBLEM_STARTS_WITH = ''

SAME_IMAGE_MAX_RMS = 0.15
SAME_DIFF_RMS_MAX = 0.001
SAME_INCREMENTAL_DIFF_STD = 0.0002

def two_means(data):
    # return two clusters
    k = 2
    centriods = np.array([np.min(data), np.max(data)])
    # print('centriods', centriods)

    while True:
        cluster = [[], []]
        for v in data:
            dis = np.abs(np.array([v] * k) - centriods)
            cluster[np.argmin(dis)].append(v)
        new_centriods = [np.mean(arr) for arr in cluster]
        if np.all(new_centriods == centriods):
            # return centriods, [(np.min(cluster[i]), np.max(cluster[i])) for i in range(0, k)]
            return cluster
        centriods = new_centriods

# two_means([0,0,0,0,1,1,1,10,10,10,8,8,8,8,7,7,7])
# two_means([0,0.100,0.2,0.3,7,8,9,9.9,9.9])

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
    # log('is_same_image', rms)
    return rms <= SAME_IMAGE_MAX_RMS

# log function, will not print message if env var XH_DEBUG is not set
def log(*args):
    flag = (os.getenv('XH_DEBUG', "").lower() in ['1', 'true', 'on'])
    if flag:
        print(*args)

class ProblemImages:
    KEYS = {
        '2x2': dict(matrix='ABC', choice='123456'),
        '3x3': dict(matrix='ABCDEFGH', choice='12345678')
    }

    def __init__(self, name, type, figures):
        self.name = name
        self.type = type
        self.images = {}
        self.rms_sum = 0

        read_image = lambda k: Image.open(figures.get(k).visualFilename)

        self.keys = keys = ProblemImages.KEYS[self.type]

        self.matrix_keys = keys['matrix']
        self.choice_keys = keys['choice']

        for k in keys['matrix'] + keys['choice']:
            self.images[k] = read_image(k)

        data = []
        keys = self.images.keys()
        for (i1, k1) in enumerate(keys):
            for (i2, k2) in enumerate(keys):
                if i1 < i2 and k1 in 'ABCDEFGHI' and k2 in 'ABCDEFGHI12345678':
                    v1 = self.images[k1]
                    v2 = self.images[k2]
                    data.append(rms_diff(v1, v2))

        self.rms_data = np.array(data)

        cluster = two_means(self.rms_data)
        cluster_range = [(np.min(cluster[i]), np.max(cluster[i])) for i in range(0, 2)]
        # log(cluster_range)

        # self.image_equal_threshold = np.min([SAME_IMAGE_MAX_RMS, np.mean([cluster_range[0][1], cluster_range[1][0]])])
        self.image_equal_threshold = SAME_IMAGE_MAX_RMS
        log(self.image_equal_threshold)



    def image_equal(self, image0, image1):
        rms = rms_diff(image0, image1)
        self.rms_sum += rms
        # log('self.image_equal', rms)
        return rms <= self.image_equal_threshold

    def sum_rms_all_pairs(self, pairs, predicate, debug=False):
        self.rms_sum = 0 # clear

        for pair in pairs:
            image0 = self.images[pair[0]]
            image1 = self.images[pair[1]]
            if not predicate(image0, image1, self.image_equal):
                return 100000

        return self.rms_sum

    def check_diff_similarity_all_pairs(self, pairs):
        diffs = []
        for pair in pairs:
            image0 = self.images[pair[0]]
            image1 = self.images[pair[1]]
            diffs.append(ImageChops.subtract(image0, image1, 2.0, 128))

        for (i0, d0) in enumerate(diffs):
            for (i1, d1) in enumerate(diffs):
                if i0 < i1:
                    rms = rms_histogram(d0, d1)
                    if rms >= SAME_DIFF_RMS_MAX:
                        return False
        
        return True

    def check_incremental_diff_directional(self, lines):

        diff_in_line = []
        for line in lines:
            diffs = []
            for (i, pair) in enumerate(line):
                image0 = self.images[pair[0]]
                image1 = self.images[pair[1]]
                diffs.append(ImageChops.subtract(image0, image1, 2.0, 128))
            diff_in_line.append(diffs)

        rms_in_line = [rms_histogram(line[0], line[1]) for line in diff_in_line]

        return np.std(np.array(rms_in_line)) < SAME_INCREMENTAL_DIFF_STD

    def check_incremental_coverage_diff_directional(self, lines):
        log('lines', lines)
        diff_in_line = []

        coverage = lambda img: sum([i * val for i, val in enumerate(img.histogram())])

        for line in lines:
            diffs = []
            for (i, pair) in enumerate(line):
                image0 = self.images[pair[0]]
                image1 = self.images[pair[1]]
                diffs.append(coverage(image1) - coverage(image0))
            diff_in_line.append(diffs)

        ratio = lambda a, b: a/float(b+a) if a + b != 0 else 1

        ratio_in_line = [ratio(line[0], line[1]) for line in diff_in_line]
        log('ratio_in_line', ratio_in_line)

        std = np.std(np.array(ratio_in_line))
        log('std', std)
        return  std < 0.2


    def check_all_pairs(self, pairs, predicate, debug=False):
        # return True only if every pair satisfies
        for pair in pairs:
            image0 = self.images[pair[0]]
            image1 = self.images[pair[1]]
            if pair == 'E2' or pair == 'E3' or pair == 'E4':
                log('false positive pair', pair)
            if not predicate(image0, image1, self.image_equal):
                if debug:
                    log('outlier pair', pair)
                return False

        return True

def shift_pair(problem_type, direction, offset):
    matrix = {
        'row': {
            '2x2': ['AB', 'C?'],
            '3x3': ['ABC', 'DEF', 'GH?']
        },
        'column': {
            '2x2': ['AC', 'B?'],
            '3x3': ['ADG', 'BEH', 'CF?']
        }
    }

    m = matrix[direction][problem_type]
    offset = offset % len(m)

    for row in range(1, len(m)):
        shift = (offset * row) % len(m)
        m[row] = m[row][shift:] + m[row][:shift]

    pairs = []
    for row in range(1, len(m)):
        for col in range(0, len(m)):
            pairs.append(m[row-1][col] + m[row][col])

    return pairs

class LocalPatternChecker:
    ALL_IDENTICAL_PAIRS = {
        '2x2': ['AB', 'BC', 'C?'],
        '3x3': ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'H?']
    }
    
    IMAGE_TRANSITIONS = [
        # lambda image0, image1: image0 == image1,
        lambda image0, image1, equal=is_same_image: equal(image0, image1),
        lambda image0, image1, equal=is_same_image: equal(image0.transpose(Image.ROTATE_90), image1),
        lambda image0, image1, equal=is_same_image: equal(image0.transpose(Image.ROTATE_180), image1),
        lambda image0, image1, equal=is_same_image: equal(image0.transpose(Image.ROTATE_270), image1)
    ]

    # Check if the entire matrix has certian local transition pattern
    def __init__(self, ProblemImages):
        self.problem = ProblemImages

    def check_preset(self):
        return self.check_all_identical() + self.check_directional_transition() + self.check_image_diff_similarity() + self.check_image_incremental_diff_similarity()

    def check_all_identical(self):
        pairs = LocalPatternChecker.ALL_IDENTICAL_PAIRS[self.problem.type]
        pred = LocalPatternChecker.IMAGE_TRANSITIONS[0]

        result = [self.problem.check_all_pairs([p.replace('?', c) for p in pairs], pred) for c in self.problem.choice_keys]
        score = np.array([1 if b else 0 for b in result])
        return score

    def check_directional_transition(self):
        max_shift = {
            '2x2': 2,
            '3x3': 3
        }

        score_acct = np.zeros(len(self.problem.choice_keys))
        for pred in LocalPatternChecker.IMAGE_TRANSITIONS:
            log('======= pred =======')
            for dir in ['row', 'column']:
                for offset in range(0, max_shift[self.problem.type]):
                    pairs = shift_pair(self.problem.type, dir, offset)
                    # debug = 'AE' in pairs and 'E?' in pairs
                    # result = [self.problem.check_all_pairs([p.replace('?', c) for p in pairs], pred, debug) for c in self.problem.choice_keys]
                    # score = np.array([1 if b else 0 for b in result])
                    result = np.array([self.problem.sum_rms_all_pairs([p.replace('?', c) for p in pairs], pred) for c in self.problem.choice_keys])
                    score = 1 / (result + 1)
                    log('dir={}, offset={}, pairs={}'.format(dir, offset, pairs), score)
                    score_acct += score

        return score_acct

    def check_image_diff_similarity(self):

        score_acct = np.zeros(len(self.problem.choice_keys))

        for dir in ['row', 'column']:
            pairs = shift_pair(self.problem.type, dir, 0)
            result = np.array([self.problem.check_diff_similarity_all_pairs([p.replace('?', c) for p in pairs]) for c in self.problem.choice_keys])
            score = np.array([1 if b else 0 for b in result])
            log('dir={}, pairs={}'.format(dir, pairs), score)
            score_acct += score

        return score_acct

    def check_image_incremental_diff_similarity(self):

        score_acct = np.zeros(len(self.problem.choice_keys))

        if self.problem.type == '2x2':
            return score_acct

        for check in [self.problem.check_incremental_diff_directional, self.problem.check_incremental_coverage_diff_directional]:
            for dir in ['row', 'column']:
                pairs = shift_pair(self.problem.type, dir, 0)
                pairs_in_line = [[pairs[i], pairs[i+3]] for i in [0, 1, 2]]
                result = np.array([check([[p.replace('?', c) for p in line] for line in pairs_in_line]) for c in self.problem.choice_keys])
                score = np.array([1 if b else 0 for b in result])
                log('dir={}, pairs={}'.format(dir, pairs), score)

                score_acct += score

        return score_acct

class GlobalPatternChecker:
    # Check if the entire matrix, viewed as one picture, has certain pattern
    PAIRS = {
        # A B
        # C ?
        '2x2': [
            # For PREDS [0] to [3]
            ['AB', 'C?'],
            ['AC', 'B?'],
            ['AA', 'BC', '??'],
            ['BB', 'A?', 'C?']
        ],
        # A B C
        # D E F
        # G H ?
        '3x3': [
            # For PREDS [0] to [3]
            ['AC', 'BB', 'DF', 'EE', 'G?', 'HH'],
            ['AG', 'DD', 'BH', 'EE', 'C?', 'FF'],
            ['AA', 'BD', 'CG', 'FH', 'EE', '??'],
            ['CC', 'EE', 'GG', 'BF', 'DH', 'A?']
        ]
    }

    PREDS = [
        # flip horizontally
        lambda image0, image1, equal=is_same_image: is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT), image1),
        # flip vertically
        lambda image0, image1, equal=is_same_image: is_same_image(image0.transpose(Image.FLIP_TOP_BOTTOM), image1),
        # flip diagnoally (axis: top-left to bottom-right)
        lambda image0, image1, equal=is_same_image: is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270), image1),
        # flip diagnoally (axis: top-right to bottom-left)
        lambda image0, image1, equal=is_same_image: is_same_image(image0.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90), image1)
    ]

    def __init__(self, ProblemImages):
        self.problem = ProblemImages

    def check_preset(self):
        pair_arry = GlobalPatternChecker.PAIRS[self.problem.type]
        preds = GlobalPatternChecker.PREDS

        score_acc = np.zeros(len(self.problem.choice_keys))

        for (pairs, pred) in zip(pair_arry, preds):
            result = [self.problem.check_all_pairs([p.replace('?', c) for p in pairs], pred) for c in self.problem.choice_keys]
            score = np.array([1 if b else 0 for b in result])
            log(score)
            score_acc += score

        return score_acc

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().

    def __init__(self):
        pass

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
        # skip problems that does not match
        if not problem.name.startswith(PROBLEM_STARTS_WITH):
            return -1

        log('{}: {}'.format(problem.name, problem.problemType))

        p = ProblemImages(problem.name, problem.problemType, problem.figures)

        return self.FindTopMatch(p)

    def FindTopMatch(self, ProblemImages):
        # Do global pattern checking
        g = GlobalPatternChecker(ProblemImages)
        global_scores = g.check_preset()
        log('global_score', global_scores)

        # Do local pattern checking
        l = LocalPatternChecker(ProblemImages)
        local_scores = l.check_preset()
        log('local_score', local_scores)

        sum = np.array(global_scores) + np.array(local_scores)

        top = np.amax(sum)
        if top == 0:
            return -1

        return np.argmax(sum) + 1

    def get_problem(self, problem, filename):
        p = ProblemImages(problem.name, problem.problemType, problem.figures)
        return p