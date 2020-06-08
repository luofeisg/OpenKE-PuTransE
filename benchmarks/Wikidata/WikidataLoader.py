from bs4 import BeautifulSoup
import re
from random import sample
from pathlib import Path
import hashlib
# from urllib2 import urlopen # Python 2
from urllib.request import urlopen  # Python 3
import bz2
import datetime
from html import unescape
import json
from qwikidata.entity import WikidataItem
from lxml import etree
import xml.etree.cElementTree as ET
from datetime import datetime


# def download_file(url, file):
#     response = urlopen(url)
#     CHUNK = 16 * 1024
#     with open(file, 'wb') as f:
#         while True:
#             chunk = response.read(CHUNK)
#             if not chunk:
#                 break
#             f.write(chunk)
#
# wikidata_dump_date = 20200501
# wikidata_url = 'https://dumps.wikimedia.org'
#
# response = urlopen(wikidata_url + '/wikidatawiki/' + str(wikidata_dump_date))
# soup = BeautifulSoup(response)
#
# # Load checksums of Wikidata dumps
# #-- Set checksum file name
# checksum_file_name = 'wikidatawiki-{}-md5sums.txt'.format(wikidata_dump_date)
# checksum_file = Path.cwd() / checksum_file_name
#
# #-- Download checksum list if not already done
# if not checksum_file.exists():
#     checksum_el = soup.find('a', href=re.compile(r'[\s\S]*md5sums*\.txt$$'))
#     checksum_file_link = wikidata_url + checksum_el.get('href')
#     download_file(checksum_file_link, checksum_file_name)
#
# #-- Load hashes out of ifle
# file_hash_dict = {}
# with checksum_file.open() as file:
#     for line in file:
#         hash, filename = line.split()
#         file_hash_dict[filename] = hash
#
# # Load Wikidata history dumps
# #-- Get relevant filenames and URLs of relevant dumps
# link_elements = soup.find_all('a', href = re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$'))
# dumps = [{'filename': element.getText(), 'url': element.get('href')} for element in link_elements]
#
# #-- download files and verify hashes
# for dump in sample(dumps, 1):
#     filename = dump['filename']
#     uri = dump['url']
#     download_file('https://dumps.wikimedia.org/' + uri, filename)
#
#     if file_hash_dict[filename] == hashlib.md5(open(filename,'rb').read()).hexdigest():
#         print('File {} downloaded successfully'.format(filename))
#     else:
#         print('Downloaded File {} has wrong md5-hash'.format(filename))
#
#
# # Traverse dumps
# for i in Path.cwd().iterdir():
#     if 'pages-meta-history' in i.name:
#
#         with bz2.open(filename) as xml_file:
#             coords = etree.parse(xml_file).getroot()
#             coords_list = []
#             for coord in coords:
#                 this = {}
#                 for child in coord.getchildren():
#                     this[child.tag] = child.text
#                     coords_list.append(this)
#
#
# filename = 'wikidatawiki-20200501-pages-meta-history27.xml-p56934357p56934729.bz2'
# with bz2.open(filename) as xml_file:
#     coords = etree.parse(xml_file).getroot()
#     coords_list = []
#     for coord in coords:
#         this = {}
#         for child in coord.getchildren():
#             this[child.tag] = child.text
#             coords_list.append(this)

def process_claims(item_id, claims):
    statements = []
    for property, statement in claims.items():
        mainsnak = claims[property][0]['mainsnak']
        if mainsnak['snaktype'] == 'value':
            if mainsnak['datavalue']['type'] == 'wikibase-entityid':
                if 'id' in mainsnak['datavalue']['value']:
                    object = mainsnak['datavalue']['value']['id']
                    statements.append((item_id, property, object))
            # else:
            #     print(mainsnak['datavalue']['type'])
        # else:
        #     print(mainsnak['snaktype']) # 'somevalue', 'novalue'

    return statements


    # 'mainsnak': {
    #     'snaktype': 'value',
    #     'property': 'P107',
    #     'hash': '5ad0e8cd324540512b927b581b5ec523db0b91fd',
    #     'datavalue': {
    #         'value': {
    #             'entity-type': 'item',
    #             'numeric-id': 215627,
    #             'id': 'Q215627'
    #         },
    #         'type': 'wikibase-entityid'
    #     }
    # }


def create_revision_file(item_id, revision_id, timestamp, claims):
    # if not exists: Create folder to save revisions.
    # Path structure will be revision_files/[item_id]/[revision_id]
    revision_files_folder = Path.cwd()/'revision_files'
    revision_files_folder.mkdir(exist_ok = True)

    # If not exists: Create folder to store the revisions of an Wikidata Item
    item_folder = revision_files_folder / str(item_id)
    item_folder.mkdir(exist_ok = True)

    with bz2.open(item_folder/ "{}.txt.bz2".format(revision_id), "wt") as f:
        f.write('{}\n'.format(timestamp))
        for rdf_triple in claims:
            h = rdf_triple[0]
            r = rdf_triple[1]
            t = rdf_triple[2]
            f.write('{}\t{}\t{}\n'.format(h, r, t))


if __name__ == '__main__':
    filename = 'wikidatawiki-20200501-pages-meta-history8.xml-p3737834p3785858.bz2'
    print(datetime.now().strftime("%H:%M:%S"))

    with bz2.open(filename, "rt", encoding="UTF-8") as xmlf:
        print(datetime.now().strftime("%H:%M:%S"))

        page_id = None
        item_id = None

        revision_id = None
        timestamp = None
        text = None  # containing claims/ facts
        claims = None
        item_dict = None
        format = None

        is_wikidata_entity = False
        for line in xmlf:
            if line.startswith("    <title>Q"):
                item_id = line[len("    <title>"):-len("</title>\n")]
                is_wikidata_entity = True
                continue

            if is_wikidata_entity:
                if line.startswith("    <id>"):
                    page_id = line[len("    <id>"):-len("</id>\n")]
                elif line.startswith("      <id>"):
                    revision_id = line[len("      <id>"):-len("</id>\n")]
                elif line.startswith("      <timestamp>"):
                    timestamp = line[len("      <timestamp>"):-len("</timestamp>\n")]
                elif line.startswith("      <format>"):
                    format = line[len("      <format>"):-len("</format>\n")]
                elif line.startswith("      <text bytes") and format == 'application/json': #and 'deleted=\"deleted\"' not in line \:
                    # Parse Text
                    text = line[line.find('>') + 1: -len('</text>') - 1]
                    if len(text) > 0:
                        # Unescape text
                        text = unescape(text)
                        # Load as WikidataItem
                        item_dict = json.loads(text)
                        if 'type' in item_dict and 'id' in item_dict:
                            item_dict = WikidataItem(item_dict)
                            claims = item_dict._entity_dict['claims']
                            if len(claims) > 0:
                                claims = process_claims(item_id, claims)
                                create_revision_file(item_id, revision_id, timestamp, claims)
                                # a= 1
                        # Write gathered data into dict

                        # print(page_id, revision_id, line)
                elif line == "    </revision>\n":
                    revision_id = None
                    timestamp = None
                    format = None
                    text = None
                    item_dict = None
                    claims = None

            if line == "  </page>\n":
                revision_id = None
                timestamp = None
                format = None
                text = None
                item_dict = None
                claims = None
                page_id = None
                item_id = None
                is_wikidata_entity = False

        print(datetime.now().strftime("%H:%M:%S"))
