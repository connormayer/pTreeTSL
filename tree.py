import pyparsing as pp
import random
from collections import defaultdict
from dataclasses import dataclass

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

# Sample projection dictionaries with lexically-specific probabilities possible
default_projections = {
    'wh+': 1,
    'wh-': 1,
    'C': 0
}

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
                                                                                label = ['car'],
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
                                        features = set(['C1']),
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
                                                                                label = ['car'],
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

print(good_parsed_tree.project(set(['wh-', 'wh+', 'C1'])).check_well_formed(check_wh))
print(bad_parsed_tree.project(set(['wh-', 'wh+', 'C1'])).check_well_formed(check_wh))
