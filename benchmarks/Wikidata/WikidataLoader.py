from bs4 import BeautifulSoup
import re
from random import sample, randint
from pathlib import Path
import hashlib
# from urllib2 import urlopen # Python 2
from urllib.request import urlopen  # Python 3
import bz2
from html import unescape
import json
from qwikidata.entity import WikidataItem
from datetime import datetime

# from nasty_utils import DecompressingTextIOWrapper
from collections import Counter, defaultdict
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor


#################################### Download Wikidata Dumps ####################################

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
    # if has_valid_checksum:
    #     print('File {} downloaded successfully'.format(filename))
    # else:
    #     print('Downloaded File {} has wrong md5-hash'.format(filename))

    return has_valid_checksum


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
        dump_file = xml_dumps_folder / xml_dump_file_dict["filename"]
        has_valid_checksum = False if not dump_file.exists() else validate_file_checksum(dump_file, wikidata_dump_date)

        if dump_file.exists() and has_valid_checksum:
            # Skip file as it was already downloaded successfully
            print("Dump {} already downloaded - SKIPPED.".format(dump_file.name))
            continue
        elif dump_file.exists() and not has_valid_checksum:
            # Delete dump if it exists but has no correct checksum to download it again
            print("Dump {} exists but has no correct checksum.".format(dump_file.name))
            dump_file.unlink()
            print("Restart download of file {} at {}.".format(dump_file.name, datetime.now()))
            download_xml_dump(xml_dump_file_dict)

        if (not dump_file.exists()) and (not validate_file_checksum(dump_file, wikidata_dump_date)):
            print("Download file {} at {}.".format(dump_file.name, datetime.now()))
            download_xml_dump(xml_dump_file_dict)


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

def create_revision_file(item_id, revision_id, timestamp, claims):
    # if not exists: Create folder to save revisions.
    # Path structure will be revision_files/[item_id]/[revision_id]
    revision_files_folder = Path.cwd() / 'revision_files'
    revision_files_folder.mkdir(exist_ok=True)

    # If not exists: Create folder to store the revisions of an Wikidata Item
    item_folder = revision_files_folder / str(item_id)
    item_folder.mkdir(exist_ok=True)

    with bz2.open(item_folder / "{}.txt.bz2".format(revision_id), "wt") as f:
        f.write('{}\n'.format(timestamp))
        for rdf_triple in claims:
            h = rdf_triple[0]
            r = rdf_triple[1]
            t = rdf_triple[2]
            f.write('{}\t{}\t{}\n'.format(h, r, t))


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

    for redirect_file_log in redirects_log_folder.iterdir():
        with bz2.open(redirect_file_log, "rt", encoding="UTF-8") as f:
            for line in f:
                source_item_id, target_item_id = line.split()
                redir_dict[source_item_id] = target_item_id

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
        objct = operation[2]
        predicate = operation[1]
        op_type = "+" if operation_type == "ins" else "-"
        triple_operations_list.append([subjct, objct, predicate, op_type, rev_ts])

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

            # Extract data
            item_id = revision_dict["item_id"]
            rev_ts = revision_dict["timestamp"]
            revision_claim_list = revision_dict["claims"]

            # Compare if ts is > than last ts
            if rev_ts >= max_ts:
                max_ts = rev_ts
            else:
                print('current timestamp {} is earlier than max timestamp {} in file {}'.format(rev_ts, max_ts,
                                                                                                revision_file))
                print(
                    "If sth goes wrong here, we have to sort the revision files with respect to the ts in ascending order")

                return

            # Process claims into set of tuples
            for claim in revision_claim_list:
                # Verify that subject is item_id
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

            # Switch olds and new set
            old_claims_set = new_claims_set
            new_claims_set = set()

    return triple_operations


def save_triple_operations(item_revision_file):
    item_revision_filename = item_revision_file.name

    # Processed marker
    processed_revision_files = Path.cwd() / item_revision_file.parents[0] / "processed_revision_files"
    processed_revision_files.mkdir(exist_ok=True)
    processed_rev_marker = processed_revision_files / "{}.processed".format(item_revision_filename)

    if processed_rev_marker.exists():
        print("Revision file {} already processed - Skip file.".format(item_revision_filename))
    else:
        item_id = item_revision_filename[:item_revision_filename.find(".")]
        item_triple_operations = get_triple_operations_list(item_revision_file)

        triple_operations_folder = Path.cwd() / 'triple_operations'
        triple_operations_folder.mkdir(exist_ok=True)

        with bz2.open(triple_operations_folder / "{}.txt.bz2".format(item_id), mode="at", encoding="UTF-8") as f:
            # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
            for op in item_triple_operations:
                line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], op[4])
                f.write(line + "\n")

        # Mark revision as processed
        processed_rev_marker.touch()


def save_revision_dict_to_json_file(item_id, revision_dict, item_is_redirected):
    # We draw all revisions of an item from the first revision that contains at least one claim.
    # Since it is possible that an item is not attached to any claims in Wikidata, we exclude such cases.
    # We only track revisions with empty claim lists if the entity possessed at least one claim before.
    # In this case, all claims of the entity at hand might deleted.

    revision_files_folder = Path.cwd() / 'revision_files'
    revision_files_folder.mkdir(exist_ok=True)

    output_filename = "redirected_{}.json.bz2".format(item_id) if item_is_redirected else "{}.json.bz2".format(item_id)
    output_filepath = revision_files_folder / output_filename

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
    print("Started processing file {} at {}.".format(file.name, datetime.now().strftime("%H:%M:%S")))
    # with DecompressingTextIOWrapper(file, encoding="UTF-8", progress_bar=True) as xmlf:
    with bz2.open(file, "rt", encoding="UTF-8") as xmlf:
        print(datetime.now().strftime("%H:%M:%S"))

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
                            save_revision_dict_to_json_file(item_id, revision_dict, item_is_redirected)

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

        print(datetime.now().strftime("%H:%M:%S"))


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

    # Load dict which maps source and target items in a redirect. We use it to replace redirected entities
    # with their target items
    redir_dict = get_redirect_dict()

    with bz2.open(output_path / "compiled_triple_operations_raw.txt.bz2", "wt") as output:
        # out.write('{}\n'.format(timestamp))
        for triple_operations_log in triple_ops_path.iterdir():
            with bz2.open(triple_operations_log, mode="rt", encoding="UTF-8") as input:
                for line in input:
                    # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
                    subj, objc, pred, op_type, ts = line.split()

                    # Resolve redirects in obj
                    new_objc = redir_dict.get(objc, objc)
                    if new_objc != objc:
                        print("Redirect! Replaced item Q{} with Q{}".format(objc, new_objc))

                    output_line = "{} {} {} {} {}".format(subj, new_objc, pred, op_type, ts)
                    output.write(output_line + "\n")


def filter_compiled_triple_operations(items_filter_list, predicates_filter_list):
    compiled_triples_path = Path.cwd() / "compiled_triple_operations"
    raw_triples_file = compiled_triples_path / "compiled_triple_operations_raw.txt.bz2"

    with bz2.open(compiled_triples_path / "compiled_triple_operations_filtered.txt.bz2", "wt") as output:
        with bz2.open(raw_triples_file, mode="rt", encoding="UTF-8") as input:
            # with DecompressingTextIOWrapper(raw_triples_file, encoding="UTF-8", progress_bar=True) as input:
            for line in input:
                subj, objc, pred, op_type, ts = line.split()
                if int(subj) in items_filter_list and int(objc) in items_filter_list and int(
                        pred) in predicates_filter_list:
                    output_line = "{} {} {} {} {}".format(subj, objc, pred, op_type, ts)
                    output.write(output_line + "\n")


def read_filter_file(file):
    filter_list = []
    with open(file) as f:
        for line in f:
            id, wikidata_id, name = line.split("\t")
            filter_list.append(int(wikidata_id))

    return filter_list


def main():
    wikidata_path = Path.cwd()
    print("Current Path is {}.".format(wikidata_path))

    wikidata_dump_date = "20200501"
    print("Start extraction process for xml history files dumped on {}.".format(wikidata_dump_date))

    ### Download XML history dumps
    # print("Download XML history dumps")
    # download_wikidata_history_dumps(wikidata_dump_date)

    ## Extract revision information about triple
    # xml_dumps_path = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    # xml_dumps_file_pattern = re.compile(r"[\s\S]*pages-meta-history.*\.bz2$$")
    # xml_dump_file_list = [xml_dump for xml_dump in xml_dumps_path.iterdir() if
    #                       xml_dump.is_file() and xml_dumps_file_pattern.match(xml_dump.name)]
    #
    # print("Extract revision information from downloaded XML dumps...")
    # with ProcessPoolExecutor() as executor:
    #     for xml_file, _ in zip(xml_dump_file_list, executor.map(process_dump_file, xml_dump_file_list)):
    #         print('File {} has been processed successfully: {}'.format(xml_file.name, datetime.now()))

    ### Extract triple operations
    print("Save paths of extracted json.bz2 revision files into list")
    revision_files_path = wikidata_path / "revision_files"
    revision_file_pattern = re.compile(r"Q.*\.json\.bz2$$")
    json_revision_files = [rev_file for rev_file in revision_files_path.iterdir() if
                           revision_file_pattern.match(rev_file.name)]

    print("Extract triple operations from json revision files.")
    with ProcessPoolExecutor() as executor:
        for file, _ in zip(json_revision_files, executor.map(save_triple_operations, json_revision_files)):
            print('DONE processing {} at {}'.format(file.name, datetime.now()))

    ## Compile dataset with triple operations and replace redirected items
    print("Compile triple operations to single file and resolve redirected object items")
    compile_triple_operations()

    # Filter triple operations with entity and relation list from LaCroix et al. (2020)
    print("Load lists of filtered entities and relations from LaCroix et al. (2020).")
    filtered_entitites = read_filter_file(Path.cwd() / "filters" / "entities_filtered_by_LaCroix_et_al_2020")
    filtered_relations = read_filter_file(Path.cwd() / "filters" / "predicates_filtered_by_LaCroix_et_al_2020")
    filter_compiled_triple_operations(filtered_entitites, filtered_relations)


if __name__ == '__main__':
    # pprint(get_redirect_dict())
    main()
