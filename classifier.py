import unidecode
import csv
import typing as tp
import numpy as np

HCM_SPECIAL_NUMBERS = [str(i) for i in range(1, 20)]
HCM = "hồchíminh"


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
               parent_node: tp.Union[TrieNode, None] = None, use_unidecode=False) -> tp.Union[TrieNode, None]:
        if parent_node is None:
            parent_node = self.root
        found_node = None
        for node_name in node_names:
            found_node = parent_node.children.get(node_name, None)
            if found_node is None:
                if use_unidecode:
                    # case (hóa and hoá)
                    node_name_without_diacritics = unidecode.unidecode(node_name)
                    found_node = {
                        unidecode.unidecode(name): node
                        for name, node in parent_node.children.items()
                    }.get(node_name_without_diacritics, None)
                    if found_node is None:
                        return None
                    parent_node = parent_node.children[found_node.node_name]
                else:
                    return None
            else:
                parent_node = parent_node.children[node_name]
        return found_node


class Solution:
    def __init__(self):
        # build trie
        self.phrase_trie = None
        self.word_trie = None
        self.letter_trie = None
        self.build_trie('dataset/new_dataset.csv')

    def build_trie(self, data_filepath: str):
        with open(data_filepath, mode='r', encoding='utf-8') as file:
            self.phrase_trie = Trie()
            self.letter_trie = Trie()
            csv_file = csv.reader(file)
            for line in csv_file:
                self.phrase_trie.insert([
                    [line[0], line[3]],
                    [line[1], line[4]],
                    [line[2], line[5]]
                ])
                for index in range(3):
                    letters = list(unidecode.unidecode(line[index]))
                    pairs = np.array([letters[::-1], letters[::-1]]).T.tolist()
                    self.letter_trie.insert(pairs)

    def find_trie_node(self, sticky_address, parent_node=None, use_unidecode=False) -> tp.Tuple[TrieNode, str]:
        current_word = ''
        index = -1
        shift = 0
        found_node = None
        found = False
        temp_found_word = None
        while not found and shift - index <= len(sticky_address):
            current_word += sticky_address[index - shift]
            found_letter = self.letter_trie.search(list(unidecode.unidecode(current_word)))
            index -= 1
            if found_letter is None:
                if temp_found_word is not None:
                    found_node = temp_found_word
                    found = True
                    index += 1
                current_word = ''  # should we just trim the latest letter that makes the string not found?
                shift += 1
                index = -1
            else:
                found_word = self.phrase_trie.search(
                    [current_word[::-1]], parent_node=parent_node, use_unidecode=use_unidecode)
                if current_word[::-1] in HCM_SPECIAL_NUMBERS and found_word is not None:
                    # check case overlapping numbers found i.e.
                    # district 3 found but actual the correct answer is 13
                    if len(current_word) == 1:
                        temp_found_word = found_word
                        continue
                    elif len(current_word) == 2:
                        if found_word is None:
                            index += 1
                            found_word = temp_found_word
                else:
                    if temp_found_word is not None:
                        found_word = temp_found_word
                        index += 1
                if found_word is not None:
                    found = True
                    found_node = found_word
        return (found_node,
                sticky_address[: index - shift + 1]
                if found_node is not None else sticky_address)

    def process(self, s: str):
        # Remove diacritics using unidecode
        sticky_address = (s.replace(',', '')
                          .replace(' ', '')
                          .replace('.', '')
                          .replace("'", '')
                          .replace("-", '').lower())
        province_node, sticky_address = self.find_trie_node(sticky_address, use_unidecode=True)
        if province_node is None:
            province_nodes = self.phrase_trie.root.children.values()
            possible_district_node_to_sticky_address = []
            for node in province_nodes:
                # TODO: check match letter cases above is subset of node.node_name
                #  to filter out some irrelevant nodes
                district_node, district_sticky_address = self.find_trie_node(
                    sticky_address, parent_node=node)
                if district_node is not None:
                    possible_district_node_to_sticky_address.append(
                        (district_node, node, district_sticky_address))
            if len(possible_district_node_to_sticky_address) == 1:
                district_node, province_node, sticky_address = possible_district_node_to_sticky_address[0]
            elif len(possible_district_node_to_sticky_address) == 2:
                second_district_node, second_province_node, _ = possible_district_node_to_sticky_address[1]
                if (second_district_node.node_name in HCM_SPECIAL_NUMBERS
                        and second_province_node.node_name == HCM):
                    district_node, province_node, _ = possible_district_node_to_sticky_address[0]
                else:
                    district_node = None
            else:
                district_node = None
        else:
            before_sticky_address = sticky_address
            district_node, sticky_address = self.find_trie_node(
                sticky_address, parent_node=province_node, use_unidecode=True)
            if district_node is None:
                sticky_address = before_sticky_address
        if district_node is not None:
            ward_node, _ = self.find_trie_node(
                sticky_address, parent_node=district_node, use_unidecode=True)
        else:
            if province_node is None:
                # retrieve district/province from reverse trie?
                ward_node = None
            else:
                district_nodes = self.phrase_trie.root.children[
                    province_node.node_name].children.values()
                possible_ward_nodes = []
                for node in district_nodes:
                    ward_node, _ = self.find_trie_node(sticky_address, parent_node=node, use_unidecode=True)
                    if ward_node is not None:
                        possible_ward_nodes.append((ward_node, node))
                if len(possible_ward_nodes) == 1:
                    ward_node, district_node = possible_ward_nodes[0]
                    if len(s1 := s.split(',')) >= 2 and not len(s1[-2].replace(' ', '')):
                        # suppose not length of district in input address, so maybe it doesn't have district
                        district_node = None
                elif len(possible_ward_nodes) == 2:
                    second_ward_node, second_district_node = possible_ward_nodes[1]
                    if second_ward_node.node_name in HCM_SPECIAL_NUMBERS:
                        ward_node, district_node = possible_ward_nodes[0]
                    # else:
                    #     ward_node = None
                    # suppose not length of district in input address, so maybe it doesn't have district
                    if len(s1 := s.split(',')) >= 2 and not len(s1[-2].replace(' ', '')):
                        # suppose not length of district in input address, so maybe it doesn't have district
                        district_node = None
                else:
                    ward_node = None
        #
        province_name = ''
        if province_node is not None:
            province_name = province_node.real_name
        district_name = ''
        if district_node is not None:
            district_name = district_node.real_name
        ward_name = ''
        if ward_node is not None:
            ward_name = ward_node.real_name

        return {
            "province": province_name,
            "district": district_name,
            "ward": ward_name,
        }


if __name__ == '__main__':
    input = "Thái Ha, HBa Vì, T.pHNội"
    solution = Solution()
    solution.process(input)
