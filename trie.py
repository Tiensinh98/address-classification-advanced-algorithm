import typing as tp

__all__ = ['TrieNode', 'Trie']


class TrieNode:
    __slots__ = ['node_name', 'real_name', 'children']

    def __init__(self, node_name=None, real_name=None):
        self.node_name = node_name
        self.real_name = real_name
        self.children = {}

    def __repr__(self):
        return f'TrieNode({self.node_name} - {self.real_name})'


class Trie:
    __slots = ['root']

    def __init__(self):
        self.root = TrieNode(node_name='root', real_name='Root')

    def insert(self, insert_data: list):
        temp_root = self.root
        for data in insert_data:
            node_name, real_name = data
            if node_name not in temp_root.children:
                new_node = TrieNode(node_name=node_name, real_name=real_name)
                temp_root.children[node_name] = new_node
            temp_root = temp_root.children[node_name]

    @staticmethod
    def get_node_by_similarity(input_name, current_node: TrieNode = None):
        provinces = current_node.children
        max_ratio = 0.0
        found_node = None
        for name, node in provinces.items():
            if abs(len(name) - len(input_name)) < 3:
                ratio = similarity(input_name, name)
                if ratio >= max(0.65, max_ratio):
                    max_ratio = ratio
                    found_node = node
        return found_node, max_ratio

    def search(self, node_names: list,
               parent_node: tp.Union[TrieNode, None] = None, use_similarity=False) -> tp.Union[TrieNode, None]:
        if parent_node is None:
            parent_node = self.root
        found_node = None
        for node_name in node_names:
            found_node = parent_node.children.get(node_name, None)
            if found_node is None:
                if use_similarity:
                    # compute similarity ratio between two strings
                    found_node, _ = Trie.get_node_by_similarity(node_name, current_node=parent_node)
                    if found_node is None:
                        return None
                    parent_node = parent_node.children[found_node.node_name]
                else:
                    return None
            else:
                parent_node = parent_node.children[node_name]
        return found_node


def levenshtein_distance(s1, s2):
    # Create a matrix to store the distances
    matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Initialize the first row and column
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j

    # Fill in the matrix
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,       # deletion
                               matrix[i][j - 1] + 1,       # insertion
                               matrix[i - 1][j - 1] + cost)  # substitution

    # Return the bottom-right cell of the matrix
    return matrix[len(s1)][len(s2)]


def similarity(s1, s2):
    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 1.0
    else:
        return 1.0 - levenshtein_distance(s1, s2) / max_length
