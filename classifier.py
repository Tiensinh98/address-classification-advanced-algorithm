import unidecode
import csv
import typing as tp
import numpy as np

HCM_SPECIAL_NUMBERS = [str(i) for i in range(1, 20)]
HCM = "hồchíminh"
SPECIAL_KEYS = ["tỉnh", "huyện", "xã",
                "phường", "thịtrấn", "khuphố", "thànhphố"]
SPECIAL_ADDRESSES = {"tphcm": "hồchíminh", "hcm": "hồchíminh"}


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


class Solution:
    def __init__(self):
        # build trie
        self.phrase_trie = None
        self.letter_trie = None
        self.build_trie('dataset/new_dataset.csv')

    def build_trie(self, data_filepath: str):
        with open(data_filepath, mode='r', encoding='utf-8') as file:
            self.phrase_trie = Trie()
            self.letter_trie = Trie()
            csv_file = csv.reader(file)
            for line in csv_file:
                self.phrase_trie.insert([
                    [unidecode.unidecode(line[0]), line[3]],
                    [unidecode.unidecode(line[1]), line[4]],
                    [unidecode.unidecode(line[2]), line[5]]
                ])
                for index in range(3):
                    letters = list(unidecode.unidecode(line[index]))
                    pairs = np.array([letters[::-1], letters[::-1]]).T.tolist()
                    self.letter_trie.insert(pairs)

    def find_trie_node(self, sticky_address, parent_node=None, use_similarity=False) -> tp.Tuple[TrieNode, str]:
        if use_similarity:
            found_node = self.phrase_trie.search(
                [sticky_address], parent_node=parent_node, use_similarity=True)
            return found_node, sticky_address if found_node is None else ''

        current_word = ''
        index = -1
        shift = 0
        found_node = None
        found = False
        temp_found_word = None
        while not found and shift - index <= len(sticky_address):
            current_word += sticky_address[index - shift]
            if current_word[::-1] in SPECIAL_KEYS:
                current_word = ''
                index = -1
                shift += 1
                continue
            found_letter = self.letter_trie.search(list(unidecode.unidecode(current_word)))
            index -= 1
            if found_letter is None:
                if temp_found_word is not None:
                    found_node = temp_found_word
                    found = True
                    index += 1
                else:
                    current_word = ''  # should we just trim the latest letter that makes the string not found?
                    shift += 1
                    index = -1
            else:
                found_word = self.phrase_trie.search(
                    [current_word[::-1]], parent_node=parent_node)
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
        return found_node, sticky_address[: index - shift + 1] if found_node is not None else sticky_address

    def find_district_node(self, sticky_address: str, province_node: TrieNode = None, use_similarity=False):
        if province_node is None:
            province_nodes = self.phrase_trie.root.children.values()
            possible_district_node_to_sticky_address = []
            for node in province_nodes:
                # TODO: check match letter cases above is subset of node.node_name
                #  to filter out some irrelevant nodes
                district_node, district_sticky_address = self.find_trie_node(
                    sticky_address, parent_node=node, use_similarity=use_similarity)
                if district_node is not None:
                    possible_district_node_to_sticky_address.append(
                        (district_node, node, district_sticky_address))
            if len(possible_district_node_to_sticky_address) == 0:
                district_node = None
            elif len(possible_district_node_to_sticky_address) == 1:
                district_node, province_node, sticky_address = possible_district_node_to_sticky_address[0]
            elif len(possible_district_node_to_sticky_address) == 2:
                second_district_node, second_province_node, _ = possible_district_node_to_sticky_address[1]
                if (second_district_node.node_name in HCM_SPECIAL_NUMBERS
                        and second_province_node.node_name == HCM):
                    district_node, province_node, sticky_address = possible_district_node_to_sticky_address[0]
                else:
                    district_node, province_node, sticky_address = possible_district_node_to_sticky_address[0]
            else:
                district_node, province_node, sticky_address = possible_district_node_to_sticky_address[0]
        else:
            district_node, sticky_address = self.find_trie_node(
                sticky_address, parent_node=province_node, use_similarity=use_similarity)

        return district_node, province_node, sticky_address

    def find_ward_node(self, sticky_address: str,
                       district_node: TrieNode = None, province_node=None, use_similarity=False):
        if district_node is not None:
            ward_node, _ = self.find_trie_node(
                sticky_address, parent_node=district_node, use_similarity=use_similarity)
        else:
            if province_node is None:
                # retrieve district/province from reverse trie?
                ward_node = None
            else:
                district_nodes = self.phrase_trie.root.children[
                    province_node.node_name].children.values()
                possible_ward_nodes = []
                for node in district_nodes:
                    ward_node, _ = self.find_trie_node(
                        sticky_address, parent_node=node)
                    if ward_node is not None:
                        possible_ward_nodes.append((ward_node, node))
                if len(possible_ward_nodes) == 0:
                    ward_node = None
                elif len(possible_ward_nodes) == 1:
                    ward_node, district_node = possible_ward_nodes[0]
                elif len(possible_ward_nodes) == 2:
                    second_ward_node, second_district_node = possible_ward_nodes[1]
                    if second_ward_node.node_name in HCM_SPECIAL_NUMBERS:
                        ward_node, district_node = possible_ward_nodes[0]
                    else:
                        ward_node, district_node = possible_ward_nodes[0]  # apply similarity here
                else:
                    ward_node, district_node = possible_ward_nodes[0]
        return ward_node, district_node

    def find_province_district_ward_node(self, segment_addresses, sticky_address, clean_address, use_similarity=False):
        initial_province_address = remove_integer_from_string(segment_addresses[-1])
        initial_province_node = None
        if len(segment_addresses) >= 1:
            if not len(initial_province_address):
                province_node = None
            else:
                province_node, sticky_address_left = self.find_trie_node(
                    initial_province_address, use_similarity=use_similarity)
                initial_province_node = province_node
                sticky_address = ''.join(segment_addresses[:-1]) + sticky_address_left
        else:
            province_node = None

        initial_district_node = None
        if len(segment_addresses) >= 2:
            initial_district_address = segment_addresses[-2]
            if not len(initial_district_address):
                district_node = None
            else:
                if len(segment_addresses[-1]) > 0:
                    district_node, province_node, sticky_address_left = self.find_district_node(
                        initial_district_address, province_node=province_node, use_similarity=use_similarity)
                else:
                    district_node, _, sticky_address_left = self.find_district_node(
                        initial_district_address, province_node=province_node, use_similarity=use_similarity)
                sticky_address = ''.join(segment_addresses[:-2]) + sticky_address_left
        else:
            district_node, province_node, sticky_address = self.find_district_node(
                sticky_address, province_node=province_node)
            initial_district_node = district_node

        if len(segment_addresses) >= 3:
            initial_ward_address = segment_addresses[-3]
            if not len(initial_ward_address):
                ward_node = None
            else:
                if len(segment_addresses[-2]) > 0:
                    ward_node, district_node = self.find_ward_node(
                        sticky_address, district_node, province_node, use_similarity=use_similarity)
                else:
                    ward_node, _ = self.find_ward_node(
                        sticky_address, district_node, province_node, use_similarity=use_similarity)
        else:
            ward_node, district_node = self.find_ward_node(
                sticky_address, district_node, province_node, use_similarity=use_similarity)

            if ward_node is None:
                # another attempt
                if province_node is not None and district_node is not None:
                    ward_node, _ = self.find_ward_node(
                        sticky_address, district_node, province_node, use_similarity=True)

            current_province_name = ''
            if province_node is not None:
                current_province_name = province_node.node_name
            current_district_name = ''
            if district_node is not None:
                current_district_name = district_node.node_name
            current_ward_name = ''
            if ward_node is not None:
                current_ward_name = ward_node.node_name
            if len(clean_address) - len(current_ward_name) \
                    - len(current_district_name) - len(current_province_name) < 0:
                if initial_province_node is None:
                    province_node = None
                if initial_district_node is None:
                    district_node = None

        return province_node, district_node, ward_node

    def process(self, s: str):
        # Remove diacritics using unidecode
        clean_address = (s.replace(' ', '')
                          .replace('.', '')
                          .replace("'", '')
                          .replace("-", '').lower())
        for key in SPECIAL_KEYS:
            clean_address = clean_address.replace(key, '')
        clean_address = unidecode.unidecode(clean_address)
        sticky_address = clean_address.replace(',', '')
        segment_addresses = clean_address.split(',')

        province_node, district_node, ward_node = self.find_province_district_ward_node(
            segment_addresses, sticky_address, clean_address)

        output_nodes = [province_node, district_node, ward_node]
        if output_nodes.count(None) > 1:
            province_node, district_node, ward_node = self.find_province_district_ward_node(
                segment_addresses, sticky_address, clean_address, use_similarity=True)

        province_name = ''
        if province_node is not None:
            province_name = province_node.real_name
        district_name = ''
        if district_node is not None:
            district_name = district_node.real_name
        ward_name = ''
        if ward_node is not None:
            ward_name = ward_node.real_name

        # hard-coded for case Nông Trường Việt Lâm :)
        if sticky_address.find("nongtruong") != -1:
            ward_name = "Nông Trường " + ward_name

        return {
            "province": province_name,
            "district": district_name,
            "ward": ward_name,
        }


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


def remove_integer_from_string(s: str, excluded_numbers=None):
    if excluded_numbers is None:
        excluded_numbers = np.arange(10)
    ns = ''
    excluded_numbers = [str(i) for i in excluded_numbers]
    for si in list(s):
        if si in excluded_numbers:
            continue
        ns += si

    return ns


if __name__ == '__main__':
    input = "F Tân bình  dĩ An "
    solution = Solution()
    output = solution.process(input)
    print(output)
