from dataclasses import dataclass
import random

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

    def __str__(self):
        child_string = ''
        if self.children:
            child_string = ":\n<{}>".format(
                ', '.join(child.__str__() for child in self.children)
            )
        return "{}[{}]{}".format(
            self.label, 
            self.features,
            child_string
        )

    def __repr__(self):
        return self.__str__()

    def project(self, target_features):
        if target_features.intersection(self.features):
            return [
                Tree(
                    label = self.label,
                    features = self.features,
                    children = [
                        projected_child
                        for child in self.children
                        for projected_child in child.project(target_features)
                    ]
                )]
        else:
            return [
                projected_child
                for child in self.children
                for projected_child in child.project(target_features)
            ]

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

# TODO: Implement grammar
# - we already have projection tier T
# - we need legal children

test_tree = Tree(
        label = 'Root',
        features = set(['S']),
        children = [
            Tree(
                label = 'Charlie',
                features = set(['NP']),
            ),
            Tree(
                label = 'eats',
                features = set(['VP']),
                children = [
                    Tree(
                        label = 'cake',
                        features = set(['N']),
                    )
                ]
            )
        ]
    )

projection = test_tree.project(set(['NP']))
print(test_tree)
print(projection)
breakpoint()