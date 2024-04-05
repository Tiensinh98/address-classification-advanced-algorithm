import unidecode


class Solution:
    def __init__(self):
        # list province, district, ward for private test, do not change for any reason
        self.province_path = 'list_province.txt'
        self.district_path = 'list_district.txt'
        self.ward_path = 'list_ward.txt'

        # build trie
        self.build_trie()

    def build_trie(self):
        return

    def process(self, s: str):
        # Remove diacritics using unidecode
        address_without_diacritics = unidecode.unidecode(s)
        sticky_address = (address_without_diacritics
                          .replace(',', '')
                          .replace(' ', '')
                          .replace('.', ''))
        print(sticky_address)
        return {
            "province": "",
            "district": "",
            "ward": "",
        }
