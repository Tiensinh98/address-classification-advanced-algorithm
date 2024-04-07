import unidecode
import csv
import typing as tp
import numpy as np


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

    def search(self, node_names: list,
               parent_node: tp.Union[TrieNode, None] = None,
               use_missing_strategy=False) -> tp.Union[TrieNode, None]:
        if parent_node is None:
            parent_node = self.root
        found_node = None
        for node_name in node_names:
            found_node = parent_node.children.get(node_name, None)
            if found_node is None:
                return None
            parent_node = parent_node.children[node_name]
        return found_node


class Solution:
    def __init__(self):
        # build trie
        self.word_trie = None
        self.letter_trie = None
        self.build_trie('dataset/dataset.csv')

    def build_trie(self, data_filepath: str):
        with open(data_filepath, mode='r', encoding='utf-8') as file:
            self.word_trie = Trie()
            self.letter_trie = Trie()
            csv_file = csv.reader(file)
            for line in csv_file:
                self.word_trie.insert([
                    [line[0], line[3]],
                    [line[1], line[4]],
                    [line[2], line[5]]
                ])
                for tup in [(0, 3), (1, 4), (2, 5)]:
                    letters0 = list(line[tup[0]])
                    # letters1 = list((line[tup[1]].replace(' ', '')))
                    pairs = np.array([letters0[::-1], letters0[::-1]]).T.tolist()
                    self.letter_trie.insert(pairs)

    def find_trie_node(self, sticky_address, parent_node=None) -> tp.Tuple[TrieNode, str]:
        current_word = ''
        index = -1
        shift = 0
        found_node = None
        found = False
        while not found:
            current_word += sticky_address[index - shift]
            found_letter = self.letter_trie.search(list(current_word))
            index -= 1
            if found_letter is None:
                current_word = ''
                shift += 1
                index = -1
            else:
                found_word = self.word_trie.search([current_word[::-1]], parent_node=parent_node)
                if found_word is not None:
                    found = True
                    found_node = found_word
        return found_node, sticky_address[: index - shift + 1]

    def process(self, s: str):
        # Remove diacritics using unidecode
        print('Input address', s)
        address_without_diacritics = unidecode.unidecode(s)
        sticky_address = (address_without_diacritics
                          .replace(',', '')
                          .replace(' ', '')
                          .replace('.', '')
                          .replace("'", '').lower())
        province_node, sticky_address = self.find_trie_node(sticky_address)
        province_name = ''
        use_missing_strategy = True
        if province_node is not None:
            province_name = province_node.real_name
            use_missing_strategy = False
        district_node, sticky_address = self.find_trie_node(sticky_address, parent_node=province_node)
        district_name = ''
        if district_node is not None:
            district_name = district_node.real_name
        ward_node, _ = self.find_trie_node(sticky_address, parent_node=district_node)
        ward_name = ''
        if ward_node is not None:
            ward_name = ward_node.real_name

        return {
            "province": province_name,
            "district": district_name,
            "ward": ward_name,
        }
