import pyparsing as pp
import random
import csv
from numpy.random import rand
from scipy.optimize import minimize
from collections import defaultdict
from dataclasses import dataclass
import warnings

@dataclass
class Word:
    form: str
    features: set

class Tree():
    def __init__(self, label, features, children=None):
        # Name of node
        self.label = label
        # Set of strings
        self.features = features
        # List of trees
        if not children:
            self.children = []
        else:
            self.children = children

    def __str__(self, depth=1):
        """
        This returns a printable string, not the string used for TSL
        """
        child_string = ''
        tabs = ''.join(['  '] * depth)

        if self.children:
            child_string = '\n{}{}'.format(
                tabs,
                '\n{}'.format(tabs).join(
                child.__str__(depth + 1) for child in self.children
            )
        )
        return "{}{}\n{}".format(
            self.label, 
            self.features,
            child_string
        )

    def __repr__(self):
        return self.__str__()

    def get_child_string(self):
        """
        This returns the string representation of the node's children.
        Not currently used, but could be helpful if you want to implement
        the SL-2 functions as string functions over child strings.
        """
        return ' '.join('{}{}'.format(child.label, child.features) 
            for child in self.children
        )

    def project(self, target_features):
        """
        Kicks off recursive process for projecting a tree
        """
        return self.project_helper(target_features)[0]

    def project_helper(self, target_features):
        """
        Recursively projects a tree
        """
        if target_features.intersection(self.features):
            return [
                Tree(
                    label = self.label,
                    features = self.features,
                    children = [
                        projected_child
                        for child in self.children
                        for projected_child in child.project_helper(target_features)
                    ]
                )]
        else:
            return [
                projected_child
                for child in self.children
                for projected_child in child.project_helper(target_features)
            ]

    def check_well_formed(self, sl_function):
        """
        Checks whether a tree is well formed given an SL function.
        """
        return sl_function(self) and all(child.check_well_formed(sl_function) for child in self.children)

    def count_child_features(self, feature):
        """
        Helper function that counts the number of children that
        contain some feature.
        """
        return sum(feature in child.features for child in self.children)

    def get_probs(self, feature_dict: dict):
        '''
        :param feature_dict: default dictionary of feature (str): probabilities (float)
        :return: probability (float)
        likely to be tweaked
        '''

        if len(set(feature_dict[self.label]).intersection(self.features))>1:
            warnings.warn("Multiple matching features.")

        for feature, prob in feature_dict[self.label].items():
            if feature in self.features:
                return prob
        return 0



    @classmethod
    def from_str(cls, string, feature_dict):
        """
        Reads in a string representation of a tree and constructs a tree
        object. feature_dict is used to map node names to features.
        """
        lpar = pp.Suppress('(')
        rpar = pp.Suppress(')')
        comma = pp.Suppress(',')
        name = pp.Word(pp.alphanums + "_")
        expr = pp.Forward()

        expr <<= (
            pp.Group(name + lpar + rpar) | 
            pp.Group(name + lpar + expr + pp.ZeroOrMore(comma + expr) + rpar)
        )

        parse = expr.parse_string(string)

        return cls.build_tree_from_parse(parse[0], feature_dict)

    @classmethod
    def build_tree_from_parse(cls, parse, feature_dict):
        """
        Recursively builds a tree from a pyparsing parse
        """
        parent, *children = parse
        features = feature_dict[parent]

        if not children:
            tree = Tree(
                label = parent,
                features = features
            )

        else:
            child_trees = [
                cls.build_tree_from_parse(child, feature_dict) for child in children
            ]
            tree = Tree(
                label = parent,
                features = features,
                children = child_trees
            )

        return tree

    def p_project(self, feature_probs, label_probs):
        # assumes all probabilities 0 < p < 1
        # and assumes only one relevant feature or label will be in the matrix at one time
        '''
        :param feature_probs: dictionary of features and probability to project
        :param label_probs: dictionary of labels and probability to project
        :return: Tree()
        '''

        if self.label in label_probs:
            x = random.random()
            if x < label_probs[self.label]:
                return [
                    Tree(
                        label=self.label,
                        features=self.features,
                        children=[
                            projected_child
                            for child in self.children
                            for projected_child in child.p_project(feature_probs, label_probs)
                        ]
                    )]

        for feature in self.features:
            if feature in feature_probs:
                x = random.random()
                if x < feature_probs[feature]:
                    return [
                        Tree(
                            label = self.label,
                            features = self.features,
                            children = [
                                projected_child
                                for child in self.children
                                for projected_child in child.p_project(feature_probs, label_probs)
                            ]
                        )]
        else:
            return [
                projected_child
                for child in self.children
                for projected_child in child.p_project(feature_probs, label_probs)
            ]


class SL2_Grammar:
    def __init__(self, functions: list, feature_dict: dict):
        '''
        :param functions: list of functions to check projected trees with
        :param feature_dict: (maybe here, maybe in trees?) default dictionary of probabilities
        may need to be edited with changes to feature representation and projection
        '''

        self.functions = functions
        self.feature_dict = feature_dict

    def is_grammatical(self, tree: Tree):
        return all([tree.check_well_formed(f) for f in self.functions])

    # under construction, adapting from Connor's pTSL paper
    def projection_p(self, tree: Tree, feature_dict: dict):
        '''
        :param tree: Tree
        :param feature_dict: dictionary of features and projection probabilities
        :return: List(Tuple(Tree, float)), list of projections and their probabilities
        '''
        # base case
        if not tree.children:
            prob = tree.get_probs(feature_dict)
            return [(proj, prob) for proj, prob in [([tree], prob), ([], 1-prob)] if prob != 0]

        # probability of being projected, function to be tweaked
        prob = tree.get_probs(feature_dict)
        sub_projections = [self.projection_p(child, feature_dict) for child in tree.children]
        possible_children = SL2_Grammar.child_product(sub_projections)

        new_children = []
        for children, val in possible_children:
            # projections including node and all possible projected children
            new_children.append(([Tree(tree.label, tree.features, children=children)], prob * val))
            # projections without node
            new_children.append((children, (1 - prob) * val))
        return new_children

    def evaluate_proj(self, proj_probs, params, corpus_probs):
        '''
        :param proj_probs: Tuple(float) probabilities in tuple equal length(params)
        :param params: List(Str) Keys for feature_dict
        :param corpus_probs: Tuple(Tree, float) trees and their acceptability/grammaticality probability
        :return:
        '''
        for i, param in enumerate(params):
            self.feature_dict[param] = proj_probs[i]

        sse = 0
        for tree, p in corpus_probs:
            sse += (self.p_grammatical(tree, self.feature_dict) - p)**2

        return sse

    def train(self, corpus_file, free_params):
        '''
        :param corpus_file: Str, location of corpus_file
        :param free_params: List(Str), dictionary keys from feature_dict whose probabilities will be optimized
        :return:
        '''

        corpus_probs = self.read_corpus_file(corpus_file, True)

        # create bounds
        # instead of limiting bound for fixed value, I removed it completely from the parameter
        bounds = [(0, 1) for _ in range(len(free_params))]

        # randomly initialize parameter - this will be the input
        proj_probs = rand(len(free_params))
        # run the minimize function
        proj_res = minimize(self.evaluate_proj,
                            proj_probs,
                            bounds=bounds,
                            method='L-BFGS-B',
                            args=(free_params, corpus_probs))

    def p_grammatical(self, tree: Tree, feature_dict: dict):
        '''
        :param tree: Tree to check
        :param feature_dict: Features to project
        :return: Probability tree is grammatical under this instantiation of grammar
        Will need rewriting, must correct case where top node does not project, but need to clarify how to treat top
        node
        '''
        return sum([prob for proj, prob in self.projection_p(tree, feature_dict) if self.is_grammatical(*proj)])

    @staticmethod
    def child_product(child_projections):
        '''
        Returns all possible combinations of child projections
        :param child_projections: list of tuples List(Tuple(Tree, float)) of children projections and probs
        :return: List(Tuple(List(Tree), float)) new list of (children, probability) tuples of possible children the
        parent node has to worry about and their probability
        '''

        first, *rest = child_projections
        # base case, return list
        if not rest:
            return first
        projections_powerset = []
        for r_projection in SL2_Grammar.projection_powerset(rest):
            # for all the other projections in the recursive call 'projection_powerset(rest)'
            for f_projection in first:
                # for all the projections currently being evaluated 'first'
                # make a new projection/probability pair which is the two lists appended and their probs multiplied
                projections_powerset.append((f_projection[0]+r_projection[0], f_projection[1]*r_projection[1]))
        # remove 0 prob projections to prevent wasteful memory usage
        return [(projection, prob) for projection, prob in projections_powerset if prob != 0]


def read_corpus_file(corpus_file):
    '''
    :param corpus_file: string specifying location of .csv file
    :return: list of (Tree, probability) tuples
    '''
    with open(corpus_file) as file:
        reader = csv.reader(file)
        corpus = [(row[0], row[1]) for row in reader]

    return corpus


# Sample projection dictionaries with lexically-specific probabilities possible

default_projections = defaultdict(lambda: 0)

default_projections.update({
    'wh+': 1,
    'wh-': 1,
    'C': 0
})

project_dict = defaultdict(lambda: default_projections)
project_dict['because'] = {
    'C': 0.4
}

def check_wh(tree):
    """
    SL-2 function for checking for wh feature violations
    """
    if 'wh+' in tree.features:
        return tree.count_child_features('wh-') == 1
    else:
        return tree.count_child_features('wh-') == 0

good_tree = Tree(
        label = 'did',
        features = set(['C', 'wh+']),
        children = [
            Tree(
                label = 'eT',
                features = set(['T', 'nom+']),
                children = [
                    Tree(
                        label = 'ev',
                        features = set(['v']),
                        children = [
                            Tree(
                                label = 'John',
                                features = set(['D', 'nom-']),
                            ),
                            Tree(
                                label = 'complain',
                                features = set(['V']),
                                children = [
                                    Tree(
                                        label = 'that',
                                        features = set(['C']),
                                        children = [

                                            Tree(
                                                label = 'eT',
                                                features = set(['T', 'nom+']),
                                                children = [
                                                    Tree(
                                                        label = 'ev',
                                                        features = ['v'],
                                                        children = [
                                                            Tree(
                                                                label = 'Mary',
                                                                features = set(['D', 'nom-']),
                                                            ),
                                                            Tree(
                                                                label = 'bought',
                                                                features = set(['V']),
                                                                children = [
                                                                    Tree(
                                                                        label = 'which',
                                                                        features = set(['D', 'wh-']),
                                                                        children = [
                                                                            Tree(
                                                                                label = 'car',
                                                                                features = set(['N'])
                                                                            )
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]  
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

bad_tree = Tree(
        label = 'did',
        features = set(['C', 'wh+']),
        children = [
            Tree(
                label = 'eps',
                features = set(['T', 'nom+']),
                children = [
                    Tree(
                        label = 'eps',
                        features = set(['v']),
                        children = [
                            Tree(
                                label = 'John',
                                features = set(['D', 'nom-']),
                            ),
                            Tree(
                                label = 'complain',
                                features = set(['V']),
                                children = [
                                    Tree(
                                        label = 'because',
                                        features = set(['C']),
                                        children = [

                                            Tree(
                                                label = 'eps',
                                                features = set(['T', 'nom+']),
                                                children = [
                                                    Tree(
                                                        label = 'eps',
                                                        features = ['v'],
                                                        children = [
                                                            Tree(
                                                                label = 'Mary',
                                                                features = set(['D', 'nom-']),
                                                            ),
                                                            Tree(
                                                                label = 'bought',
                                                                features = set(['V']),
                                                                children = [
                                                                    Tree(
                                                                        label = 'which',
                                                                        features = set(['D', 'wh-']),
                                                                        children = [
                                                                            Tree(
                                                                                label = 'car',
                                                                                features = set(['N'])
                                                                            )
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]  
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

good_projection = good_tree.project(set(['wh-', 'wh+', 'C1']))
print(good_projection)
print(good_projection.check_well_formed(check_wh))

bad_projection = bad_tree.project(set(['wh-', 'wh+', 'C1']))
print(bad_projection)
print(bad_projection.check_well_formed(check_wh))

feature_dict = {
    'did': set(['C', 'wh+']),
    'eT': set(['T', 'nom+']),
    'ev': set(['v']),
    'John': set(['D', 'nom-']),
    'complain': set(['V']),
    'that': set(['C']),
    'because': set(['C1']),
    'Mary': set(['D', 'nom-']),
    'bought': set(['V']),
    'which': set(['D', 'wh-']),
    'car': set(['N'])
}

good_tree_str = """
    did(
        eT(
            ev(
                John(),
                complain(
                    that(
                        eT(
                            ev(
                                Mary(),
                                bought(
                                    which(
                                        car()
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )"""

bad_tree_str = """
    did(
        eT(
            ev(
                John(),
                complain(
                    because(
                        eT(
                            ev(
                                Mary(),
                                bought(
                                    which(
                                        car()
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )"""
good_parsed_tree = Tree.from_str(good_tree_str, feature_dict)
bad_parsed_tree = Tree.from_str(bad_tree_str, feature_dict)
grammar = Grammar([check_wh], feature_dict)

print(good_parsed_tree.project(set(['wh-', 'wh+', 'C1'])).check_well_formed(check_wh))
print(bad_parsed_tree.project(set(['wh-', 'wh+', 'C1'])).check_well_formed(check_wh))
print(grammar.p_grammatical(bad_tree, project_dict))
print(grammar.p_grammatical(good_tree, project_dict))

