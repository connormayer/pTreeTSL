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