'''
MIT License

Copyright (c) 2020 Rashid Lafraie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
import os
from bs4 import BeautifulSoup
import re
from pathlib import Path
import hashlib
from urllib.request import urlopen
import bz2
from html import unescape
import json
from datetime import datetime
import numpy as np
import multiprocessing as mp
from nasty_utils import DecompressingTextIOWrapper
from concurrent.futures import ProcessPoolExecutor
import operator


def download_file(url, file):
    # Helper function to download large files in chunks #
    # Used to download the xml history dumps            #

    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(file, 'wb') as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)


def process_checksum_file(checksum_file, wikidata_dump_date):
    checksum_folder = checksum_file.parents[0]

    # XML history file pattern
    file_pattern = re.compile(r"[\s\S]*pages-meta-history.*\.bz2$$")

    with checksum_file.open() as file:
        for line in file:
            hash, filename = line.split()
            if file_pattern.match(filename):
                checksum_filename = filename + "_checksum.txt"
                with open(checksum_folder / checksum_filename, mode="at", encoding="UTF-8") as out:
                    out.write(hash)


def get_wikidata_dump_filelist(wikidata_dump_date=20200501):
    # Get list of xml history files and their URLs. #
    # Download and process checksums in order to    #
    # validate the downloaded files                 #

    wikidata_url = 'https://dumps.wikimedia.org'

    response = urlopen(wikidata_url + '/wikidatawiki/' + str(wikidata_dump_date))
    soup = BeautifulSoup(response, "html.parser")

    # Load checksums of Wikidata dumps
    checksum_folder = Path.cwd() / "checksums_{}".format(wikidata_dump_date)
    checksum_folder.mkdir(exist_ok=True)

    checksum_file = checksum_folder / 'wikidatawiki-{}-md5sums.txt'.format(wikidata_dump_date)
    if not checksum_file.exists():
        checksum_el = soup.find('a', href=re.compile(r'[\s\S]*md5sums*\.txt$$'))
        checksum_file_link = wikidata_url + checksum_el.get('href')
        download_file(checksum_file_link, checksum_file)
        process_checksum_file(checksum_file, wikidata_dump_date)

    # -- Get relevant filenames and URLs of relevant dumps
    link_elements = soup.find_all('a', href=re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$'))
    dumps = [{"filename": element.getText(),
              "url": element.get('href'),
              "dumpdate": wikidata_dump_date}
             for element in link_elements]

    return dumps


def validate_file_checksum(file, wikidata_dump_date):
    filename = file.name

    # Load checksum of xml dump
    checksum_filename = filename + "_checksum.txt"
    checksum_file = Path.cwd() / "checksums_{}".format(wikidata_dump_date) / checksum_filename
    with open(checksum_file, mode="rt", encoding="UTF-8") as f:
        checksum = f.read()

    # Validate downloaded file
    has_valid_checksum = checksum == hashlib.md5(open(file, 'rb').read()).hexdigest()
    if has_valid_checksum:
        print('File {} downloaded successfully'.format(filename))
    else:
        sys.exit('Downloaded File {} has wrong md5-hash!'.format(filename))


def download_xml_dump(file_download_dict):
    filename = file_download_dict["filename"]
    wikidata_dump_date = file_download_dict["dumpdate"]
    uri = file_download_dict["url"]

    xml_dumps_folder = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    xml_dumps_folder.mkdir(exist_ok=True)
    xml_dump_file = xml_dumps_folder / filename

    download_file('https://dumps.wikimedia.org/' + uri, xml_dump_file)
    validate_file_checksum(xml_dump_file, wikidata_dump_date)


def download_wikidata_history_dumps(wikidata_dump_date):
    ### Create output folders to store the dumps
    xml_dumps_folder = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    xml_dumps_folder.mkdir(exist_ok=True)
    downloaded_marker_folder = xml_dumps_folder / "downloaded_dumps"
    downloaded_marker_folder.mkdir(exist_ok=True)

    ### Get list of XML dumps with urls to download and process and download files
    xml_dump_file_list = get_wikidata_dump_filelist(wikidata_dump_date)
    # xml_dump_file_list = [file for file in xml_dump_file_list if
    #                       file["filename"] == "wikidatawiki-20200501-pages-meta-history27.xml-p56934357p56934729.bz2"]
    for xml_dump_file_dict in xml_dump_file_list:
        dump_file_name = xml_dump_file_dict["filename"]
        dump_file = xml_dumps_folder / dump_file_name
        downloaded_marker_file = downloaded_marker_folder / "{}.downloaded".format(dump_file_name)
        # has_valid_checksum = False if not dump_file.exists() else validate_file_checksum(dump_file, wikidata_dump_date)

        if downloaded_marker_file.exists():
            # Skip file as it was already downloaded successfully
            print("Dump {} already downloaded - SKIPPED.".format(dump_file_name))
            continue
        else:
            # If dump file exists but donwload marker not exists it means that previous download has been aborted
            # or had no correct checksum
            if dump_file.exists():
                # Delete dump if it exists but has no correct checksum to download it again
                print("Dump {} exists but was previously aborted.".format(dump_file_name))
                dump_file.unlink()
                print("Restart download of file {} at {}.".format(dump_file_name, datetime.now()))
            else:
                print("Download file {} at {}.".format(dump_file_name, datetime.now()))

            download_xml_dump(xml_dump_file_dict)
            # Create 'successfully downloaded' marker
            downloaded_marker_file.touch()


# def download_wikidata_history_dumps(filename, url, wikidata_dump_date=20200501):
#     # Load checksums of Wikidata dumps
#     checksum_file_name = 'wikidatawiki-{}-md5sums.txt'.format(wikidata_dump_date)
#     checksum_file = Path.cwd() / checksum_file_name
#
#     # -- Download checksum list if not already done
#     if not checksum_file.exists():
#         checksum_el = soup.find('a', href=re.compile(r'[\s\S]*md5sums*\.txt$$'))
#         checksum_file_link = wikidata_url + checksum_el.get('href')
#         download_file(checksum_file_link, checksum_file_name)
#
#     # -- Load hashes out of file
#     file_hash_dict = {}
#     with checksum_file.open() as file:
#         for line in file:
#             hash, filename = line.split()
#             file_hash_dict[filename] = hash
#
#     # -- download files and verify hashes
#     for dump in dumps:
#         filename = dump['filename']
#         uri = dump['url']
#         download_file('https://dumps.wikimedia.org/' + uri, filename)
#
#         if file_hash_dict[filename] == hashlib.md5(open(filename, 'rb').read()).hexdigest():
#             print('File {} downloaded successfully'.format(filename))
#         else:
#             print('Downloaded File {} has wrong md5-hash'.format(filename))

# Return list with downloaded files
# return [dump["filename"] for dump in dumps]


#################################### Extract and save xml dump information ####################################
def create_revision_dict(item_id, revision_id, timestamp, claim_triple_list):
    revision_dict = {
        "item_id": item_id,
        "revision": revision_id,
        "timestamp": timestamp,
        "claims": claim_triple_list
    }

    return revision_dict


def create_redirect_entry(filename, source_item_id, target_item_id):
    # Create folder to store information about redirects
    redirects_log_folder = Path.cwd() / 'redirects'
    redirects_log_folder.mkdir(exist_ok=True)

    with bz2.open("redirects/{}_redirected_items.txt.bz2".format(filename), mode="at", encoding="UTF-8") as f:
        f.write("{} {}\n".format(source_item_id[1:], target_item_id[1:]))


def get_redirect_dict():
    redirects_log_folder = Path.cwd() / 'redirects'
    redir_dict = {}

    print("Load redirects from {} at {}.".format(redirects_log_folder, datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    if redirects_log_folder.exists():
        for redirect_file_log in redirects_log_folder.iterdir():
            with bz2.open(redirect_file_log, "rt", encoding="UTF-8") as f:
                for line in f:
                    source_item_id, target_item_id = line.split()
                    redir_dict[source_item_id] = target_item_id

    print("Finished loading redirects at {}.".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    print("Counted {} redirects.".format(len(redir_dict)))
    return redir_dict


def process_redirect(filename, source_item_id, target_item_id):
    # Delete last revisions of redirected entity where all claims have been deleted
    strip_revision_file_of_redirected_item(source_item_id)
    # Store information about redirect in file
    create_redirect_entry(filename, source_item_id, target_item_id)


def extract_triple_operations(new_claims_set, old_claims_set, rev_ts, operation_type="ins"):
    triple_operations_list = []

    # Get insert operations (new_set - old_set)
    triple_operations = new_claims_set - old_claims_set if operation_type == "ins" else old_claims_set - new_claims_set
    for operation in triple_operations:
        subjct = operation[0]
        predicte = operation[1]
        objct = operation[2]
        op_type = "+" if operation_type == "ins" else "-"
        triple_operations_list.append([subjct, objct, predicte, op_type, rev_ts])

    return triple_operations_list


def get_triple_operations_list(revision_file):
    triple_operations = []
    old_claims_set = set()
    new_claims_set = set()
    max_ts = "0001-01-01T01:00:00Z"

    with bz2.open(revision_file, "rt", encoding="UTF-8") as jsonf:
        for line in jsonf:
            # Read revision dict from line
            revision_dict = json.loads(line)
            revision_file_name = revision_file.name
            # Extract data
            item_id = revision_dict["item_id"]
            rev_ts = revision_dict["timestamp"]
            revision_claim_list = revision_dict["claims"]

            # Compare if ts is > than last ts
            if rev_ts >= max_ts:
                max_ts = rev_ts
            else:
                print('current timestamp {} is earlier than max timestamp {} in file {}'.format(rev_ts, max_ts,
                                                                                                revision_file_name))
                print(
                    "If sth goes wrong here, we have to sort the revision files with respect to the ts in ascending order")
                buggy_revision_files_folder = Path.cwd() / revision_file.parents[1] / "buggy_revision_files"
                buggy_revision_files_folder.mkdir(exist_ok=True)
                buggy_revision_files_dump_subfolder = buggy_revision_files_folder / revision_file.parents[0].name
                buggy_revision_files_dump_subfolder.mkdir(exist_ok=True)
                buggy_revision_file_marker = buggy_revision_files_dump_subfolder / "{}.buggy".format(revision_file_name)
                buggy_revision_file_marker.touch()

            # Process claims into set of tuples
            for claim in revision_claim_list:
                # Verify that claim head (claim[0]) represents item_id
                if claim[0] == int(item_id[1:]):
                    new_claims_set.add(tuple(claim))
                else:
                    print("Subject {} != item_id {} for triple in file {}: ".format(claim[0], item_id, revision_file))
                    print("--> ", claim)
                    return

            # Get insert operations (new_set - old_set) and encode them as "<subject> <object> <predicate> <rev_ts> <operation_type>"
            insert_operations = extract_triple_operations(new_claims_set, old_claims_set, rev_ts, operation_type="ins")
            triple_operations.extend(insert_operations)

            # Get delete operations (old_set - new_set) and encode them as "<subject> <object> <predicate> <rev_ts> <operation_type>"
            delete_operations = extract_triple_operations(new_claims_set, old_claims_set, rev_ts, operation_type="del")
            triple_operations.extend(delete_operations)

            # Switch set new_claim_set to old_claim_set for processing the next revision in jsonf
            old_claims_set = new_claims_set
            new_claims_set = set()

    return triple_operations


def extract_revision_folders_triple_operations(rev_folder):
    print(
        "Extract triple operations from folder {} at {}.".format(rev_folder.name, datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    item_revision_file_list = [file for file in rev_folder.iterdir() if
                               file.is_file() and not file.name.startswith("redirected_")]
    for item_revision_file in item_revision_file_list:
        save_triple_operations(item_revision_file)


def save_triple_operations(item_revision_file):
    item_revision_filename = item_revision_file.name

    # Processed marker
    processed_revision_files = Path.cwd() / item_revision_file.parents[1].name / "processed_revision_files"
    processed_revision_files.mkdir(exist_ok=True)

    processed_revision_files_dump_subfolder = processed_revision_files / item_revision_file.parents[0].name
    processed_revision_files_dump_subfolder.mkdir(exist_ok=True)

    processed_rev_marker = processed_revision_files_dump_subfolder / "{}.processed".format(item_revision_filename)

    if processed_rev_marker.exists():
        print("Revision file {} already processed - Skip file.".format(item_revision_filename))
    else:
        # Cut out item id QXXX from filename pattern <filedump_name>_QXXX.json.bz2
        item_id = item_revision_filename[item_revision_filename.find("Q"):item_revision_filename.find(".json")]
        item_triple_operations = get_triple_operations_list(item_revision_file)

        triple_operations_folder = Path.cwd() / 'triple_operations'
        triple_operations_folder.mkdir(exist_ok=True)

        dump_subfolder = triple_operations_folder / item_revision_file.parents[0].name
        dump_subfolder.mkdir(exist_ok=True)

        output_filename = "{}.txt.bz2".format(item_id)
        output_filepath = dump_subfolder / output_filename

        with bz2.open(output_filepath, mode="wt", encoding="UTF-8") as f:
            # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
            for op in item_triple_operations:
                line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], op[4])
                f.write(line + "\n")

        # Mark revision as processed
        processed_rev_marker.touch()


def save_revision_dict_to_json_file(dump_file_name, item_id, revision_dict, item_is_redirected):
    # We draw all revisions of an item from the first revision that contains at least one claim.
    # Since it is possible that an item is not attached to any claims in Wikidata, we exclude such cases.
    # We only track revisions with empty claim lists if the entity possessed at least one claim before.
    # In this case, all claims of the entity at hand might deleted.

    revision_files_folder = Path.cwd() / "revision_files"
    revision_files_folder.mkdir(exist_ok=True)

    dump_subfolder = revision_files_folder / '{}'.format(dump_file_name)
    dump_subfolder.mkdir(exist_ok=True)

    output_filename = "redirected_{}_{}.json.bz2".format(dump_file_name, item_id) \
        if item_is_redirected else "{}_{}.json.bz2".format(dump_file_name, item_id)
    output_filepath = dump_subfolder / output_filename

    # Catch cases in which the claim list is empty and no revisions for an entity have been stored before
    if (not output_filepath.exists()) and len(revision_dict["claims"]) == 0:
        return

    else:
        revision_json = json.dumps(revision_dict)
        with bz2.open(output_filepath, mode="at", encoding="UTF-8") as f:
            f.write(revision_json + "\n")

    # {
    #     "revision": "1010281398",
    #     "timestamp": "2019-09-08T18:43:46Z",
    #     "claims": [
    #         [
    #             <item_id>         "3964154",
    #             <predicate_id>    "17",
    #             <object_id>       "159"
    #         ],
    #         [
    #             "Q3964154",
    #             "P131",
    #             "Q2246"
    #         ],
    #         [
    #             "Q3964154",
    #             "P31",
    #             "Q13626398"
    #         ]
    #     ]
    # }


def strip_revision_file_of_redirected_item(source_item_id):
    # check if file exists
    source_file_path = Path.cwd() / "revision_files" / '{}.json.bz2'.format(source_item_id)
    if source_file_path.exists():
        # Cut last n revision with no claims

        lines = []
        last_filled_revision_index = None

        with bz2.open(source_file_path, "rt", encoding="UTF-8") as f:
            # Iterate lines reversely
            for idx, line in enumerate(f):
                revision_dict = json.loads(line)

                if len(revision_dict["claims"]) > 0:
                    lines.append(revision_dict)
                    last_filled_revision_index = idx

        # Overwrite file with new lines
        lines = map(json.dumps, lines[:last_filled_revision_index])
        bz2.open(source_file_path, "wt", encoding="UTF-8").writelines(lines)


def print_out_bz2_file(filename):
    with bz2.open(filename, "rt", encoding="UTF-8") as f:
        for line in enumerate(f):
            print(line)


def get_truthy_claims_list(item_dict):
    statements = []

    if len(item_dict['claims']) > 0:

        item_id = item_dict["id"]
        claim_list_dict = item_dict["claims"]

        for proprty, claim_list in claim_list_dict.items():
            preferred_statements = []
            normal_statements = []

            for claim_dict in claim_list:
                rank = claim_dict["rank"].lower()  # rank is always in ['deprecated' | 'preferred' | 'normal']
                if rank != "deprecated":
                    mainsnak = claim_dict["mainsnak"]
                    snak_type = mainsnak["snaktype"].lower()

                    # In case "datavalue" is not contained in mainsnak--> mainsnak['snaktype'] = ['somevalue' | 'novalue']
                    if snak_type == "value" and "datavalue" in mainsnak:

                        # check if object is Wikidata Entity
                        # (In case mainsnak['datavalue']['type'] != 'wikibase-entityid'
                        # --> mainsnak['datavalue']['type'] in
                        # [ 'string'
                        # | 'monolingualtext'
                        # | 'time'
                        # | 'quantity'
                        # | 'globecoordinate'])
                        mainsnak_type = mainsnak['datavalue']['type']
                        if mainsnak_type == 'wikibase-entityid':

                            objct_dict = mainsnak['datavalue']['value']
                            objct_type = objct_dict['entity-type']

                            # check if object_type is 'item'
                            # (Otherwise it is a 'property')
                            if objct_type == "item":
                                # object_id = "Q{}".format(object_dict["numeric-id"])
                                triple = (int(item_id[1:]), int(proprty[1:]), int(objct_dict["numeric-id"]))

                                if rank == "preferred":
                                    preferred_statements.append(triple)
                                elif rank == "normal":
                                    normal_statements.append(triple)

                                # Check if numeric_id is always like id without the prefix "Q"
                                if "id" in objct_dict and str(objct_dict["numeric-id"]) != objct_dict["id"][1:]:
                                    print("Different ids for numeric-id {} and id {}".format(
                                        objct_dict["numeric-id"],
                                        objct_dict["id"][1:]))

            statements.extend(preferred_statements) if len(preferred_statements) > 0 else statements.extend(
                normal_statements)

    return statements

    # #--------- [item_dict structure] -------------#
    # {
    #     "type": "item",
    #     "id": "Q3918736",
    #     "labels": {
    #         "ru": {
    #             "language": "ru",
    #             "value": "Седов, Валентин Васильевич"
    #         },
    #         "be": {
    #             "language": "be",
    #             "value": "Валянцін Васілевіч Сядоў"
    #         },
    #         "lv": {
    #             "language": "lv",
    #             "value": "Valentīns Sedovs"
    #         }
    #     },
    #     "descriptions": [],
    #     "aliases": {
    #         "ru": [
    #             {
    #                 "language": "ru",
    #                 "value": "Седов В. В."
    #             },
    #             {
    #                 "language": "ru",
    #                 "value": "Валентин Васильевич Седов"
    #             },
    #             {
    #                 "language": "ru",
    #                 "value": "Седов Валентин Васильевич"
    #             }
    #         ],
    #         "be": [
    #             {
    #                 "language": "be",
    #                 "value": "Валянцін Васільевіч Сядоў"
    #             },
    #             {
    #                 "language": "be",
    #                 "value": "Валянцін Сядоў"
    #             }
    #         ],
    #         "lv": [
    #             {
    #                 "language": "lv",
    #                 "value": "Sedovs"
    #             }
    #         ]
    #     },
    #     "claims": {
    #         "P107": [
    #             {
    #                 "mainsnak": {
    #                     "snaktype": "value",
    #                     "property": "P107",
    #                     "hash": "5ad0e8cd324540512b927b581b5ec523db0b91fd",
    #                     "datavalue": {
    #                         "value": {
    #                             "entity-type": "item",
    #                             "numeric-id": 215627,
    #                             "id": "Q215627"
    #                         },
    #                         "type": "wikibase-entityid"
    #                     }
    #                 },
    #                 "type": "statement",
    #                 "id": "q3918736$DBFA7AF6-46D8-45F0-B04F-CD5597FCF58E",
    #                 "rank": "normal"
    #             }
    #         ]
    #     },
    #     "sitelinks": {
    #         "ruwiki": {
    #             "site": "ruwiki",
    #             "title": "Седов, Валентин Васильевич",
    #             "badges": []
    #         },
    #         "bewiki": {
    #             "site": "bewiki",
    #             "title": "Валянцін Васілевіч Сядоў",
    #             "badges": []
    #         },
    #         "lvwiki": {
    #             "site": "lvwiki",
    #             "title": "Valentīns Sedovs",
    #             "badges": []
    #         }
    #     }
    # }

    # --------- [Claims structure] --> item_dict["claims"]
    # "claims": {
    #         "P107": [
    #             {
    #                 "mainsnak": {
    #                     "snaktype": "value",
    #                     "property": "P107",
    #                     "hash": "5ad0e8cd324540512b927b581b5ec523db0b91fd",
    #                     "datavalue": {
    #                         "value": {
    #                             "entity-type": "item",
    #                             "numeric-id": 215627,
    #                             "id": "Q215627"
    #                         },
    #                         "type": "wikibase-entityid"
    #                     }
    #                 },
    #                 "type": "statement",
    #                 "id": "q3918736$DBFA7AF6-46D8-45F0-B04F-CD5597FCF58E",
    #                 "rank": "normal"
    #             }
    #         ]
    #     }

    # --------- [Mainsnak structure] --> claims[property_id][0]["mainsnak"]] -------------#
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


def process_xml_dump(file):
    print("Started processing file {} at {}.".format(file.name, datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    # with DecompressingTextIOWrapper(file, encoding="UTF-8", progress_bar=True) as xmlf:
    with bz2.open(file, "rt", encoding="UTF-8") as xmlf:
        print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

        item_id = None
        revision_id = None
        timestamp = None
        text = None  # containing claims/ facts
        claims = None
        item_dict = None
        format = None
        item_is_redirected = False

        for line in xmlf:
            if line.startswith("    <title>"):
                item_id = line[len("    <title>"):-len("</title>\n")]

            if line.startswith("    <redirect title") and item_id.startswith("Q"):
                redir_target_item_id = line[len("    <redirect title ="):-len("\" />\n")]
                # create redirect entry
                item_is_redirected = True
                create_redirect_entry(file.name, item_id, redir_target_item_id)

            elif line.startswith("      <id>"):
                revision_id = line[len("      <id>"):-len("</id>\n")]
            elif line.startswith("      <timestamp>"):
                timestamp = line[len("      <timestamp>"):-len("</timestamp>\n")]
            elif line.startswith("      <comment>"):
                comment = line[len("      <comment>"):-len("</comment>\n")]
            elif line.startswith("      <format>"):
                format = line[len("      <format>"):-len("</format>\n")]
            elif line.startswith("      <text bytes"):
                if format == 'application/json' and item_id and item_id.startswith("Q"):
                    # Parse Text
                    text = line[line.find('>') + 1: -len('</text>') - 1]

                    if len(text) > 0:
                        # Unescape text
                        text = unescape(text)
                        item_dict = json.loads(text)

                        # Check if item_dict contains "type" and "id" as keys
                        # --> Otherwise, keys of item_dict are either ("entity", "redirect") or ("flow-workflow")
                        if 'type' in item_dict and 'id' in item_dict:
                            # Process and store information about revisions of an Wikidata item in JSON
                            claim_triple_list = get_truthy_claims_list(item_dict)
                            revision_dict = create_revision_dict(item_id, revision_id, timestamp, claim_triple_list)
                            save_revision_dict_to_json_file(file.name, item_id, revision_dict, item_is_redirected)

                        # elif 'entity' in item_dict and 'redirect' in item_dict and item_is_redirected:
                        # source_item_id = item_dict["entity"]
                        # target_item_id = item_dict["redirect"]
                        #
                        # if target_item_id == redir_item_id:
                        #     process_redirect(file, source_item_id, target_item_id)
                        # else:
                        #     print("For source item {} target_item_id in markup \<redirect title\> is {}, but "
                        #           "target in item_dict[\"redirect\"] is {}.".format(source_item_id,
                        #                                                             redir_item_id,
                        #                                                             target_item_id))
                        # source_item_id = None
                        # target_item_id = None



            elif line == "    </revision>\n":
                revision_id = None
                timestamp = None
                format = None
                text = None
                item_dict = None
                claims = None
                claim_triple_list = None
                revision_dict = None

            if line == "  </page>\n":
                revision_id = None
                timestamp = None
                format = None
                text = None
                item_dict = None
                claims = None
                page_id = None
                item_id = None
                item_is_redirected = False
                redir_target_item_id = None
                redir_item_id = None

        print(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))


# def download_and_process_xml_dump(file_download_dict):
#     filename = file_download_dict["filename"]
#     wikidata_dump_date = file_download_dict["dumpdate"]
#
#     print("Download file {}".format(filename))
#     download_xml_dump(file_download_dict)
#
#     print("Process file {}".format(filename))
#     xml_dump_file = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date) / filename
#     process_xml_dump(xml_dump_file)
#
#     print("Delete processed file {}".format(filename))
#     xml_dump_file.unlink()


def process_dump_file(file):
    print("Process file {}\n".format(file.name))

    # Mark file as processed by creating a file with extension ".processed" in the sub folder "processed dumps" so
    # we can see if file has been already processed after restarting the long extraction process again after an
    # interruption
    processed_xml_dumps_folder = Path.cwd() / file.parents[0] / "processed_dumps"
    processed_xml_dumps_folder.mkdir(exist_ok=True)
    processed_marker = processed_xml_dumps_folder / "{}.processed".format(file.name)

    if processed_marker.exists():
        print("File {} already processed - Skip file.".format(file.name))
    else:
        process_xml_dump(file)
        processed_marker.touch()


def compile_triple_operations():
    # If not exists: Create output directory
    output_path = Path.cwd() / "compiled_triple_operations"
    output_path.mkdir(exist_ok=True)

    # Path where triple ops are stored for each item
    triple_ops_path = Path.cwd() / 'triple_operations'
    triple_ops_dump_subfolders = [fld for fld in triple_ops_path.iterdir() if
                                  fld.is_dir() and not fld.name.startswith("processed_")]

    # triple_ops_file_list = [file for file in triple_ops_path.rglob("*txt.bz2") if file.is_file()]
    # Load dict which maps source and target items in a redirect. We use it to replace redirected entities
    # with their target items
    redir_dict = get_redirect_dict()
    print("Found {} folders containing item triple operations.".format(len(triple_ops_dump_subfolders)))

    for subfolder in triple_ops_dump_subfolders:
        subfolder_triple_ops = [file for file in subfolder.iterdir() if file.is_file() and file.name.startswith("Q")]
        ("Get triple operations from {}.".format(subfolder.name))
        for triple_operations_log in subfolder_triple_ops:
            processed_triple_ops_folder = triple_ops_path / "processed_triple_operations"
            processed_triple_ops_folder.mkdir(exist_ok=True)

            processed_triple_ops_dump_subfolder = processed_triple_ops_folder / subfolder.name
            processed_triple_ops_dump_subfolder.mkdir(exist_ok=True)

            processed_triple_ops_marker = processed_triple_ops_dump_subfolder / "{}.processed".format(
                triple_operations_log.name)
            if processed_triple_ops_marker.exists():
                print("Triple operations file {} already processed - Skip file.".format(triple_operations_log.name))
            else:
                output_lines = []
                with bz2.open(triple_operations_log, mode="rt", encoding="UTF-8") as input:
                    for line in input:
                        # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
                        subj, objc, pred, op_type, ts = line.split()

                        # Resolve redirects in obj
                        new_objc = redir_dict.get(objc, objc)
                        if new_objc != objc:
                            print("Redirect! Replaced item Q{} with Q{}".format(objc, new_objc))

                        out_line = "{} {} {} {} {}\n".format(subj, new_objc, pred, op_type, ts)
                        # output.write(output_line + "\n")
                        output_lines.append(out_line)

                    # Transmit operations to file
                    with bz2.open(output_path / "compiled_triple_operations_raw.txt.bz2", mode="at",
                                  encoding="utf-8") as output:
                        output.writelines(output_lines)

                    # Create processed marker
                    processed_triple_ops_marker.touch()
        print("Finished gathering of triple extraction for folder {}.".format(subfolder.name))


def process_subfolder_triple_operations(subfolder, q, redir_dict=None, filters=None):
    subfolder_triple_ops = [file for file in subfolder.iterdir() if file.is_file() and file.name.startswith("Q")]
    print("Get triple operations from {}.".format(subfolder.name))

    # Folder for processed markers
    processed_triple_ops_fld = Path.cwd() / 'triple_operations' / "processed_triple_operations_v1"
    processed_triple_ops_fld.mkdir(exist_ok=True)

    processed_triple_ops_dump_subfld = processed_triple_ops_fld / subfolder.name
    processed_triple_ops_dump_subfld.mkdir(exist_ok=True)

    for triple_operations_log in subfolder_triple_ops:
        processed_triple_ops_marker = processed_triple_ops_dump_subfld / "{}.processed".format(
            triple_operations_log.name)

        if processed_triple_ops_marker.exists():
            print("Triple operations file {} already processed - Skip file.".format(triple_operations_log.name))
        else:
            output_lines = []
            with bz2.open(triple_operations_log, mode="rt", encoding="UTF-8") as input:
                for line in input:
                    # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
                    subj, objc, pred, op_type, ts = line.split()

                    # Resolve redirects in obj
                    if redir_dict:
                        objc = redir_dict.get(objc, objc)

                    # If filter is attached use it to only collect selected triples ops
                    if filters:
                        if not (subj in filters["filtered_entities"]
                                and objc in filters["filtered_entities"]
                                and pred in filters["filtered_relations"]) or (subj == objc):
                            continue

                    out_line = "{} {} {} {} {}\n".format(subj, objc, pred, op_type, ts)
                    output_lines.append(out_line)

                if output_lines:
                    # Transmit operations to Queue
                    q.put(output_lines)
                    # Create processed marker
                    processed_triple_ops_marker.touch()

    return ("Finished gathering of triple extraction for folder {}.".format(subfolder.name))


def writer(q, file):
    '''listens for messages on the q, writes to file. '''

    with bz2.open(file, mode="at", encoding="utf-8") as output:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            output.writelines(m)
            output.flush()


def compile_triple_operations_v1(num_cpu_cores, filters=None, resolve_redir=True):
    # If not exists: Create output directory
    output_path = Path.cwd() / "compiled_triple_operations"
    output_path.mkdir(exist_ok=True)

    # Output file
    output_file = output_path / "compiled_triple_operations_directly_filtered.txt.bz2"

    # Use Manager queue here to delegate writing into a single file from multiple jobs
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_cpu_cores)

    # Start writer process
    watcher = pool.apply_async(writer, (q, output_file))

    # Load dict which maps source and target items in a redirect. We use it to replace redirected entities
    # with their target items
    redir_dict = get_redirect_dict()

    # Path where triple ops are stored for each item
    triple_ops_path = Path.cwd() / 'triple_operations'
    triple_ops_dump_subfolders = [fld for fld in triple_ops_path.iterdir()
                                  if fld.is_dir() and not fld.name.startswith("processed_")]
    print("Found {} folders containing item triple operations.".format(len(triple_ops_dump_subfolders)))

    # Each subfolder is attached to a job
    jobs = []
    for subfolder in triple_ops_dump_subfolders:
        job = pool.apply_async(process_subfolder_triple_operations, (subfolder, q, redir_dict, filters))
        jobs.append(job)

    # Collect job results
    for job in jobs:
        result = job.get()
        print(result)

    # Kill the writer process
    q.put('kill')
    pool.close()
    pool.join()


def filter_compiled_triple_operations(items_filter_list, predicates_filter_list):
    compiled_triples_path = Path.cwd() / "compiled_triple_operations"
    raw_triples_file = compiled_triples_path / "compiled_triple_operations_raw.txt.bz2"

    with bz2.open(compiled_triples_path / "compiled_triple_operations_filtered.txt.bz2", "wt") as output:
        # with bz2.open(raw_triples_file, mode="rt", encoding="UTF-8") as input:
        with DecompressingTextIOWrapper(raw_triples_file, encoding="UTF-8", progress_bar=True) as input:
            gathered_operations = 0
            total_operations = 0
            for line in input:
                subj, objc, pred, op_type, ts = line.split()
                total_operations += 1
                if subj in items_filter_list and objc in items_filter_list and pred in predicates_filter_list:
                    output_line = "{} {} {} {} {}".format(subj, objc, pred, op_type, ts)
                    output.write(output_line + "\n")
                    gathered_operations += 1

    print(
        "Finished filtering process by selecting {} out of {} operations".format(gathered_operations, total_operations))


def read_filter_file(file):
    filter_list = []
    with open(file) as f:
        for line in f:
            id, wikidata_id, name = line.split("\t")
            # filter_list.append(int(wikidata_id))
            filter_list.append(wikidata_id)

    return filter_list


def remove_duplicates(triple_operations):
    index = 0
    consistent_triple_operations = []
    triple_state_dict = {}

    while index + 1 < len(triple_operations):
        curr_subjc, curr_objc, curr_pred, curr_op_type, curr_ts = triple_operations[index]
        curr_triple = (curr_subjc, curr_objc, curr_pred)

        next_subjc, next_objc, next_pred, next_op_type, next_ts = triple_operations[index + 1]
        next_triple = (next_subjc, next_objc, next_pred)

        # Handle duplicate triple operation (h,r,t,+,ts) --> (h,r,t,-,ts)
        if curr_triple == next_triple and curr_ts == next_ts and curr_op_type != next_op_type:
            ''' To solve the following pattern:
                -------------------------------
                2521388 1622272 106 + 2013-04-12T16:12:19Z <-> before replacing redirected object with target: 2521388 10860762 106 + 2013-04-12T16:12:19Z
                2521388 1622272 106 + 2013-06-02T12:12:01Z <-> before replacing redirected object with target: 2521388 1660508 106 + 2013-06-02T12:12:01Z
                2521388 1622272 106 - 2013-06-02T12:12:01Z <-> before replacing redirected object with target: 2521388 10860762 106 - 2013-06-02T12:12:01Z
                2521388 1622272 106 + 2013-10-17T19:24:53Z <-> before replacing redirected object with target: 2521388 1622272 106 + 2013-10-17T19:24:53Z
                2521388 1622272 106 - 2013-10-17T19:24:53Z <-> before replacing redirected object with target: 2521388 1660508 106 - 2013-10-17T19:24:53Z'''

            index += 2
            continue

        # Handle first operation for a triple
        if curr_triple not in triple_state_dict:
            if curr_op_type == "-":
                print("Invalid triple operations pattern. First operation for {} is a deletion".format(curr_triple))


        # Handle duplicate triple operation (h,r,t,+,ts) --> (h,r,t,+,ts + 1)
        elif triple_state_dict[curr_triple] == curr_op_type:
            ''' To solve the following pattern:
                    -------------------------------
                2527494 21139794 527 + 2015-11-12T13:40:26Z <-> before replacing redirected object with target: 2527494 21139794 527 + 2015-11-12T13:40:26Z 
                2527494 21139794 527 + 2015-11-12T13:40:35Z <-> before replacing redirected object with target: 2527494 21141770 527 + 2015-11-12T13:40:35Z
                2527494 21139794 527 - 2016-01-22T03:18:50Z <-> before replacing redirected object with target: 2527494 21139794 527 - 2016-01-22T03:18:50Z
                2527494 21139794 527 - 2016-01-22T03:18:52Z <-> before replacing redirected object with target: 2527494 21141770 527 - 2016-01-22T03:18:52Z'''
            index += 1
            continue

        triple_state_dict[curr_triple] = curr_op_type
        consistent_triple_operations.append(triple_operations[index])
        index += 1

    return consistent_triple_operations


# def resolve_inconsistent_triple_ops_sequence(triple_operations):
#     ''' 2521388 1622272 106 + 2013-04-12T16:12:19Z <-> before replacing redirected object with target: 2521388 10860762 106 + 2013-04-12T16:12:19Z
#         2521388 1622272 106 + 2013-06-02T12:12:01Z <-> before replacing redirected object with target: 2521388 1660508 106 + 2013-06-02T12:12:01Z
#         2521388 1622272 106 - 2013-06-02T12:12:01Z <-> before replacing redirected object with target: 2521388 10860762 106 - 2013-06-02T12:12:01Z
#         2521388 1622272 106 + 2013-10-17T19:24:53Z <-> before replacing redirected object with target: 2521388 1622272 106 + 2013-10-17T19:24:53Z
#         2521388 1622272 106 - 2013-10-17T19:24:53Z <-> before replacing redirected object with target: 2521388 1660508 106 - 2013-10-17T19:24:53Z'''
#
#     triple_state_dict = {}
#     index = 0
#     while (index < len(triple_operations) - 1):
#         curr_subjc, curr_objc, curr_pred, curr_op_type, curr_ts = triple_operations[index]
#         curr_triple = (curr_subjc, curr_objc, curr_pred)
#
#         next_subjc, next_objc, next_pred, next_op_type, next_ts = triple_operations[index + 1]
#         next_triple = (next_subjc, next_objc, next_pred)
#
#         # Handle first operation for a triple
#         if curr_triple not in triple_state_dict:
#             # Handle case if first occurence is triple deletion
#             if curr_op_type == "-":
#                 if next_triple == curr_triple and next_op_type == "+":
#                     # Switch indexes of current and next operation
#                     triple_operations[index] = (next_subjc, next_objc, next_pred, next_op_type, next_ts)
#                     triple_operations[index + 1] = (curr_subjc, curr_objc, curr_pred, curr_op_type, curr_ts)
#                     # Continue after next index
#                     triple_state_dict[curr_triple] = curr_op_type
#                     index += 2
#                     continue
#                 else:
#                     print("Invalid triple operations pattern. First operation for {} is a deletion".format(curr_triple))
#
#         # Handle subsequent occurences
#         else:
#             if curr_triple == next_triple:
#                 if triple_state_dict[curr_triple] == curr_op_type:
#                     triple_operations[index] = (next_subjc, next_objc, next_pred, next_op_type, next_ts)
#                     triple_operations[index + 1] = (curr_subjc, curr_objc, curr_pred, curr_op_type, curr_ts)
#                     # Continue after next index
#                     triple_state_dict[curr_triple] = curr_op_type
#                     index += 2
#                     continue
#
#         triple_state_dict[curr_triple] = curr_op_type
#         index += 1


def sort_filtered_triple_operations_v1(input_file_name, output_filename, compress_output=False):
    print("Load filtered triple operations.")
    compiled_triples_path = Path.cwd() / "compiled_triple_operations"
    input_file = compiled_triples_path / input_file_name

    # Get triple operations from file
    triple_operations = []
    with bz2.open(input_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            # subj, objc, pred, op_type, ts = line.split()
            triple_operations.append(line.split())

    # Sort triple operations with respect to timestamp, triple, op_type
    triple_operations = sorted(triple_operations, key=operator.itemgetter(4, 0, 1, 2, 3))
    # TEST Save sorted triple operations list in file
    # output_name = "test_sorted"
    # sorted_triple_ops_file = compiled_triples_path / "{}.txt".format(output_name)
    # with sorted_triple_ops_file.open(mode="wt", encoding="UTF-8") as f:
    #     for index, op in enumerate(triple_operations):
    #         # line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], timestamps_sorted[index])
    #         line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], op[4])
    #         f.write(line + "\n")

    # Resolve inconsistencies that emerge after replacing redirected objects with their target_id
    # - where inserts and deletes of a triples possess the same ts
    triple_operations = remove_duplicates(triple_operations)

    print("Save sorted list to file.")
    if compress_output:
        output_name = output_filename if output_filename else "compiled_triple_operations_filtered_and_sorted"
        sorted_triple_ops_file = compiled_triples_path / "{}.txt.bz2".format(output_name)
        f = bz2.open(sorted_triple_ops_file, mode="wt", encoding="UTF-8")
    else:
        output_name = output_filename if output_filename else "compiled_triple_operations_filtered_and_sorted"
        sorted_triple_ops_file = compiled_triples_path / "{}.txt".format(output_name)
        f = sorted_triple_ops_file.open(mode="wt", encoding="UTF-8")

    # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
    for index, op in enumerate(triple_operations):
        # line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], timestamps_sorted[index])
        line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], op[4])
        f.write(line + "\n")
    f.close()


def sort_filtered_triple_operations(input_file_name, output_filename):
    print("Load filtered triple operations.")
    compiled_triples_path = Path.cwd() / "compiled_triple_operations"
    input_file = compiled_triples_path / input_file_name

    triple_operations = []
    timestamps = []
    with bz2.open(input_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            subj, objc, pred, op_type, ts = line.split()
            triple_operations.append((int(subj), int(objc), int(pred), op_type))
            timestamps.append(ts)

    timestamps = np.array(timestamps)
    timestamps_sorted_indexes = timestamps.argsort().tolist()
    timestamps_sorted = timestamps[timestamps_sorted_indexes]
    triple_operations = [triple_operations[i] for i in timestamps_sorted_indexes]

    print("Save sorted list to file.")
    output_name = output_filename if output_filename else "compiled_triple_operations_filtered_and_sorted"
    sorted_triple_ops_file = compiled_triples_path / "{}.txt.bz2".format(output_name)
    with bz2.open(sorted_triple_ops_file, mode="wt", encoding="UTF-8") as f:
        # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
        for index, op in enumerate(triple_operations):
            line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], timestamps_sorted[index])
            f.write(line + "\n")


def main():
    # Obtain number of CPU cores
    num_cores_available = os.cpu_count()

    num_cores_granted = int(input(
        "There are {} CPU cores available at your system. How many of them do you want to grant to the Wikidata extraction process?".format(
            num_cores_available)))

    wikidata_dump_date = "20200501"
    wikidata_path = Path.cwd()
    print("Current Path is {}.".format(wikidata_path))

    print("Start extraction process for xml history files dumped on {}.".format(wikidata_dump_date))

    # Download XML history dumps
    print("Download XML history dumps")
    # download_wikidata_history_dumps(wikidata_dump_date)

    # Extract revision information about triple
    xml_dumps_path = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    xml_dumps_file_pattern = re.compile(r"[\s\S]*pages-meta-history.*\.bz2$$")
    xml_dump_file_list = [xml_dump for xml_dump in xml_dumps_path.iterdir() if
                          xml_dump.is_file() and xml_dumps_file_pattern.match(xml_dump.name)]

    print("Extract revision information from downloaded XML dumps...")
    with ProcessPoolExecutor(max_workers=num_cores_granted) as executor:
        for xml_file, _ in zip(xml_dump_file_list, executor.map(process_dump_file, xml_dump_file_list)):
            print('File {} has been processed successfully: {}'.format(xml_file.name, datetime.now()))

    # Extract triple operations
    print("Save paths of extracted json.bz2 revision files into list")
    revision_files_path = wikidata_path / "revision_files"
    revision_folder_pattern = re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$')
    json_revision_folder = [rev_folder for rev_folder in revision_files_path.iterdir() if rev_folder.is_dir()
                            and revision_folder_pattern.match(rev_folder.name)]

    print("Extract triple operations from json revision files.")
    with ProcessPoolExecutor(max_workers=num_cores_granted) as executor:
        for folder, _ in zip(json_revision_folder,
                             executor.map(extract_revision_folders_triple_operations, json_revision_folder)):
            print('DONE processing folder {} at {}'.format(folder.name, datetime.now()))

    # Compile dataset with triple operations and replace redirected items
    # --> Time spent: compiled_triple_operations_raw.txt.bz2: 100%|██████████| 9.56M/9.56M [1:03:12<00:00, 2.64kB/s]
    print("Compile triple operations to single file and resolve redirected object items.")
    compile_triple_operations()

    print("Load lists of filtered entities and relations from LaCroix et al. (2020).")
    filter_path = Path.cwd() / "filters"
    filtered_entities = read_filter_file(filter_path / "entities_filtered_by_LaCroix_et_al_2020")
    filtered_relations = read_filter_file(filter_path / "predicates_filtered_by_LaCroix_et_al_2020")

    print("Filter triple operations.")
    filter_compiled_triple_operations(filtered_entities, filtered_relations)

    # Filter triple operations with entity and relation list from LaCroix et al. (2020)
    # Time spent: compiled_triple_operations_raw.txt.bz2: 100%|██████████| 10.3M/10.3M [49:59<00:00, 3.61kB/s]

    sort_filtered_triple_operations(input_file_name="compiled_triple_operations_filtered.txt.bz2",
                                    output_filename="Sorted_triple_ops")


def main2():
    # Obtain number of CPU cores
    num_cores_available = os.cpu_count()

    num_cores_granted = int(input(
        "There are {} CPU cores available at your system. How many of them do you want to grant to the Wikidata extraction process?".format(
            num_cores_available)))

    wikidata_dump_date = "20200501"
    wikidata_path = Path.cwd()
    print("Current Path is {}.".format(wikidata_path))

    print("Start extraction process for xml history files dumped on {}.".format(wikidata_dump_date))

    # Download XML history dumps
    # print("Download XML history dumps")
    # download_wikidata_history_dumps(wikidata_dump_date)
    #
    # # Extract revision information about triple
    # xml_dumps_path = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    # xml_dumps_file_pattern = re.compile(r"[\s\S]*pages-meta-history.*\.bz2$$")
    # xml_dump_file_list = [xml_dump for xml_dump in xml_dumps_path.iterdir() if
    #                       xml_dump.is_file() and xml_dumps_file_pattern.match(xml_dump.name)]
    #
    # print("Extract revision information from downloaded XML dumps...")
    # with ProcessPoolExecutor(max_workers=num_cores_granted) as executor:
    #     for xml_file, _ in zip(xml_dump_file_list, executor.map(process_dump_file, xml_dump_file_list)):
    #         print('File {} has been processed successfully: {}'.format(xml_file.name, datetime.now()))
    #
    # # Extract triple operations
    # print("Save paths of extracted json.bz2 revision files into list")
    # revision_files_path = wikidata_path / "revision_files"
    # revision_folder_pattern = re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$')
    # json_revision_folder = [rev_folder for rev_folder in revision_files_path.iterdir() if rev_folder.is_dir()
    #                         and revision_folder_pattern.match(rev_folder.name)]
    #
    # # print("Extract triple operations from json revision files.")
    # with ProcessPoolExecutor(max_workers=num_cores_granted) as executor:
    #     for folder, _ in zip(json_revision_folder,
    #                          executor.map(extract_revision_folders_triple_operations, json_revision_folder)):
    #         print('DONE processing folder {} at {}'.format(folder.name, datetime.now()))

    # Compile dataset with triple operations and replace redirected items
    # Time spent: start:15:57:36 --> end: 16:30:33
    print("Compile triple operations to single file "
          "+ resolve redirected object items "
          "+ Filter entities and relations accordinng to LaCroix et al. (2020)")

    print("Load lists of filtered entities and relations from LaCroix et al. (2020).")
    filter_path = Path.cwd() / "filters"
    filters = {"filtered_entities": read_filter_file(filter_path / "entities_filtered_by_LaCroix_et_al_2020"),
               "filtered_relations": read_filter_file(filter_path / "predicates_filtered_by_LaCroix_et_al_2020")}

    print("Start compilation of triple operations at {}.".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    compile_triple_operations_v1(num_cores_granted, filters=filters, resolve_redir=True)
    print("Finish at {}.".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))

    print("Sort and save triple operations.")
    sort_filtered_triple_operations_v1(input_file_name="compiled_triple_operations_directly_filtered.txt.bz2",
                                       output_filename="compiled_triple_operations_directly_filtered_and_sorted",
                                       compress_output=True)


if __name__ == '__main__':
    main2()

