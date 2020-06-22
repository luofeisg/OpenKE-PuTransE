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

from nasty_utils import DecompressingTextIOWrapper
from collections import Counter, defaultdict
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor


#################################### Download Wikidata Dumps ####################################
def download_file(url, file):
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(file, 'wb') as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)


def download_wikidata_history_dumps(wikidata_dump_date=20200501):
    wikidata_url = 'https://dumps.wikimedia.org'

    response = urlopen(wikidata_url + '/wikidatawiki/' + str(wikidata_dump_date))
    soup = BeautifulSoup(response, "html.parser")

    # Load checksums of Wikidata dumps
    checksum_file_name = 'wikidatawiki-{}-md5sums.txt'.format(wikidata_dump_date)
    checksum_file = Path.cwd() / checksum_file_name

    # -- Download checksum list if not already done
    if not checksum_file.exists():
        checksum_el = soup.find('a', href=re.compile(r'[\s\S]*md5sums*\.txt$$'))
        checksum_file_link = wikidata_url + checksum_el.get('href')
        download_file(checksum_file_link, checksum_file_name)

    # -- Load hashes out of file
    file_hash_dict = {}
    with checksum_file.open() as file:
        for line in file:
            hash, filename = line.split()
            file_hash_dict[filename] = hash

    # Load Wikidata history dumps
    # -- Get relevant filenames and URLs of relevant dumps
    link_elements = soup.find_all('a', href=re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$'))
    dumps = [{'filename': element.getText(), 'url': element.get('href')} for element in link_elements]

    # -- download files and verify hashes
    for dump in dumps:
        filename = dump['filename']
        uri = dump['url']
        download_file('https://dumps.wikimedia.org/' + uri, filename)

        if file_hash_dict[filename] == hashlib.md5(open(filename, 'rb').read()).hexdigest():
            print('File {} downloaded successfully'.format(filename))
        else:
            print('Downloaded File {} has wrong md5-hash'.format(filename))

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
    # For redirections we store the source_item_id and dest_item_id as tuples. After finished processing and storing
    # revision information, we merge the align the growth of claims in corresponding source_items and dest_items

    # Procedure
    # 1) Load redirects into dict with dest_item_id -> [source_id_1, source_id_2, ..] or
    # maybe tuples (dest_id , source_id_1, source_id_2, ..)
    #
    # 2) Process all entity files:
    # 2.1) Create 2 sets: one in which items are affected by redirects and one in which they are not
    # 2.2) Process deletes, insertsions (deltas) of unaffected ones
    # 2.3) Process all affected item_ids
    # 2.3.1) Gather all tuples like (dest_id , source_id_1, source_id_2, ..)
    # 2.3.2) Special Handling:

    # Create folder to store information about redirects
    redirects_log_folder = Path.cwd() / 'redirects'
    redirects_log_folder.mkdir(exist_ok=True)

    with bz2.open("redirects/{}_redirected_items.txt.bz2".format(filename), mode="at", encoding="UTF-8") as f:
        f.write("{} {}\n".format(source_item_id, target_item_id))


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
        subject = operation[0]
        object = operation[2]
        predicate = operation[1]
        operation_type = "+" if operation_type == "ins" else "-"
        triple_operations_list.append([subject, object, predicate, operation_type, rev_ts])

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
                if claim[0] == item_id:
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
    if item_revision_file.exists():
        item_triple_operations = get_triple_operations_list(item_revision_file)
        item_revision_filename = item_revision_file.name
        item_id = item_revision_filename[:item_revision_filename.find(".")]
    else:
        print("No revision file {} found.".format(item_revision_file))
        return
    triple_operations_folder = Path.cwd() / 'triple_operations'
    triple_operations_folder.mkdir(exist_ok=True)

    with bz2.open(triple_operations_folder / "{}.txt.bz2".format(item_id), mode="at", encoding="UTF-8") as f:
        # triple_operation format : [subject, object, predicate, operation_type, rev_ts]
        for op in item_triple_operations:
            line = "{} {} {} {} {}".format(op[0], op[1], op[2], op[3], op[4])
            f.write(line + "\n")


def save_revision_dict_to_json_file(item_id, revision_dict):
    # For us, an entity is created if there is a first revision which contains at least one claim.
    # Since it is possible that an entity may not have any claims if its corresponding page is created, we do not track
    # such cases. This means that we only track empty claim lists if revisions with at least one claim have already
    # occurred before. In this case, all claims of the entity at hand might deleted.

    revision_files_folder = Path.cwd() / 'revision_files'
    revision_files_folder.mkdir(exist_ok=True)

    entity_file_path = revision_files_folder / "{}.json.bz2".format(item_id)

    if (not entity_file_path.exists()) and len(revision_dict["claims"]) == 0:
        return

    else:
        revision_json = json.dumps(revision_dict)
        with bz2.open("revision_files/{}.json.bz2".format(item_id), mode="at", encoding="UTF-8") as f:
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

        for property, claim_list in claim_list_dict.items():
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

                            object_dict = mainsnak['datavalue']['value']
                            object_type = object_dict['entity-type']

                            # check object_type is Item
                            # (Otherwise it is a Property)
                            if object_type == "item":
                                # object_id = "Q{}".format(object_dict["numeric-id"])
                                triple = (int(item_id[1:]), int(property[1:]), int(object_dict["numeric-id"]))

                                if rank == "preferred":
                                    preferred_statements.append(triple)
                                elif rank == "normal":
                                    normal_statements.append(triple)

                                # Check if numeric_id is always like id without the prefix "Q"
                                if "id" in object_dict and str(object_dict["numeric-id"]) != object_dict["id"][1:]:
                                    print("Different ids for numeric-id {} and id {}".format(
                                        object_dict["numeric-id"],
                                        object_dict["id"][1:]))

            statements.extend(preferred_statements) if len(preferred_statements) != 0 else statements.extend(
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


def process_xml_dump(filename):
    # mode can either be "analysis" or "extraction".
    # mode = "analysis": Traverses XML dump and calculate statistics to enable the exploration of the underlying data
    # mode = "extraction":
    print(datetime.now().strftime("%H:%M:%S"))

    with DecompressingTextIOWrapper(Path(filename), encoding="UTF-8", progress_bar=True) as xmlf:
        # with bz2.open(filename, "rt", encoding="UTF-8") as xmlf:
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

            if line.startswith("    <redirect title"):
                redir_item_id = line[len("    <redirect title ="):-len("\" />\n")]
                # create redirection entry
                item_is_redirected = True

            elif line.startswith("      <id>"):
                revision_id = line[len("      <id>"):-len("</id>\n")]
            elif line.startswith("      <timestamp>"):
                timestamp = line[len("      <timestamp>"):-len("</timestamp>\n")]
            elif line.startswith("      <comment>"):
                comment = line[len("      <comment>"):-len("</comment>\n")]
            elif line.startswith("      <format>"):
                format = line[len("      <format>"):-len("</format>\n")]
            elif line.startswith("      <text bytes"):
                if format == 'application/json':
                    # Parse Text
                    text = line[line.find('>') + 1: -len('</text>') - 1]

                    if len(text) > 0:
                        # Unescape text
                        text = unescape(text)
                        item_dict = json.loads(text)

                        # Process and store information about revisions of an Wikidata item in JSON

                        if item_id and item_id.startswith("Q"):
                            # check if item_dict contains "type" and "id" as keys
                            # --> Otherwise keys of item_dict are either ("entity", "redirect") or ("flow-workflow")
                            if 'type' in item_dict and 'id' in item_dict:
                                claim_triple_list = get_truthy_claims_list(item_dict)
                                revision_dict = create_revision_dict(item_id, revision_id, timestamp, claim_triple_list)
                                save_revision_dict_to_json_file(item_id, revision_dict)

                            elif 'entity' in item_dict and 'redirect' in item_dict and item_is_redirected:
                                source_item_id = item_dict["entity"]
                                target_item_id = item_dict["redirect"]

                                if target_item_id == redir_item_id:
                                    process_redirect(filename, source_item_id, target_item_id)
                                else:
                                    print("For source item {} target_item_id in markup \<redirect title\> is {}, but "
                                          "target in item_dict[\"redirect\"] is {}.".format(source_item_id,
                                                                                            redir_item_id,
                                                                                            target_item_id))
                                source_item_id = None
                                target_item_id = None



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
                redir_item_id = None

        print(datetime.now().strftime("%H:%M:%S"))


def main():
    print("Download history xml dumps from URL https://dumps.wikimedia.org/wikidatawiki/.")
    download_wikidata_history_dumps(wikidata_dump_date=20200501)

    print("Save paths of downloaded files into list.")
    xml_dump_file_list = [xml_dump_file for xml_dump_file in Path.cwd().glob('*pages-meta-history*')]
    xml_dump_file_list = xml_dump_file_list[:8]

    print("Extract triples from xml dumps.")
    with ProcessPoolExecutor() as executor:
        for filename, _ in zip(xml_dump_file_list, executor.map(process_xml_dump, xml_dump_file_list)):
            print('File {} has been processed succesfully: {}'.format(filename, datetime.now()))

    print("Save paths of extracted json.bz2 revision files into list")
    json_revision_files = [rev_file for rev_file in Path.cwd().glob('revision_files/*json.bz2*')]

    print("Extract triple operations from json revision files.")
    with ProcessPoolExecutor() as executor:
        for file, _ in zip(json_revision_files, executor.map(save_triple_operations, json_revision_files)):
            print('Revision file {} processed succesfully at {}'.format(file.name, datetime.now()))

    print("Process redirected entities")
    # TODO

    print("Load lists of filtered entities and relations from LaCroix et al. (2020).")
    filtered_entitites = read_filter_file(Path.cwd() / "filters" / "entities_filtered_by_LaCroix_et_al_2020")
    filtered_relations = read_filter_file(Path.cwd() / "filters" / "predicates_filtered_by_LaCroix_et_al_2020")




def read_filter_file(file):
    filter_list = []
    with open(file) as f:
        for line in f:
            id, wikidata_id, name = line.split("\t")
            filter_list.append("{}".format(wikidata_id))

    return filter_list


if __name__ == '__main__':
    # main()

    filename = 'wikidatawiki-20200501-pages-meta-history20.xml-p19981616p20061203.bz2'
    # filename = 'wikidatawiki-20200501-pages-meta-history1.xml-p1p242.bz2'
    process_xml_dump(filename)

    # open_entity_revision_json_file("revision_files/Q3918736.json.bz2")

    # filename = 'Q3918780.json.bz2'
    # filepath = Path.cwd() / 'revision_files' / filename
    # triple_list = get_triple_operations_list(filepath)
    # triple_list.append(1

    # save_triple_operations("Q3921737")
    # print_out_bz2_file(Path.cwd() / "revision_files" / "Q15.json.bz2")
    # print_out_bz2_file(Path.cwd() / "triple_operations" / "Q3921737.txt.bz2")

    #
