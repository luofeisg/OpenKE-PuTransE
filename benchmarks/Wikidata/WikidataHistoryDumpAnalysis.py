import sys
from random import randint
from pathlib import Path
from html import unescape
import json
from qwikidata.entity import WikidataItem
from datetime import datetime
from nasty_utils import DecompressingTextIOWrapper
from collections import Counter, defaultdict
from pprint import pprint


def get_xml_attribute_information(xml_markup, output="key"):
    attributes_tuple = ()

    text = xml_markup.strip()
    text = text[text.find(' '):text.find('>')].strip()
    text = text.split(" ")

    if output == "key":
        attributes_tuple = tuple(attribute[:attribute.find("=")] for attribute in text)
    elif output == "value":
        attributes_tuple = tuple(attribute[attribute.find("=") + 1:] for attribute in text)

    return attributes_tuple


def process_titles(filename):
    print(datetime.now().strftime("%H:%M:%S"))

    with DecompressingTextIOWrapper(Path(filename), encoding="UTF-8", progress_bar=True) as xmlf:
        # with bz2.open(filename, "rt", encoding="UTF-8") as xmlf:
        print(datetime.now().strftime("%H:%M:%S"))

        for line in xmlf:
            if line.startswith("    <title>"):
                item_id = line[len("    <title>"):-len("</title>\n")]
                print(item_id)


def analyse_xml_dump(filename):
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

        # Qualitative Data Analysis: Variables for qualitative data analysis
        incompatible_items_list = []
        text_bytes_execeptional_contents = []
        mainsnak_key_content_dict = defaultdict(list)
        # claim_key_content_dict = defaultdict(list)

        # Quantitative Data Analysis: Counter variables
        title_counter = Counter()
        redirections_counter = Counter()
        format_counter = Counter()
        wikidata_compatibility_counter = Counter()
        item_dict_key_counter = Counter()
        claims_counter = Counter()
        mainsnak_dict_key_counter = Counter()
        text_attribute_counter = Counter()
        text_attribute_value_correlation = defaultdict(Counter)  # defaultdict: attribute_key -> value
        mainsnak_value_entity_types_counter = Counter()
        mainsnak_datavalue_types_counter = Counter()
        mainsnak_datavalue_value_keys_counter = Counter()
        mainsnak_datavalue_value_keys_samples = defaultdict(lambda: defaultdict(list))  # keys -> item_id -> content
        mainsnak_datavalue_types_value_entity_type_cooccurences = defaultdict(Counter)
        mainsnak_datavalue_types_value_samples = defaultdict(list)
        mainsnak_snaktype_counter = Counter()
        claim_rank_types_counter = Counter()
        claim_dict_keys_counter = Counter()
        len_claim_list_counter = Counter()
        len_claim_dict_counter = Counter()

        text_attribute_text_length_correlation = defaultdict(
            Counter)  # defaultdict: attribute_key -> len(text) ([>|=] 0)
        text_attribute_claim_length_correlation = defaultdict(
            Counter)  # defaultdict: attribute_key -> len(claims) ([>|=] 0)

        item_claim_growth = defaultdict(list)

        for line in xmlf:
            if line.startswith("    <title>"):
                item_id = line[len("    <title>"):-len("</title>\n")]
                title_counter[item_id[:1]] += 1

            if line.startswith("    <redirect title"):
                redir_item_id = line[len("    <redirect title ="):-len("\" />\n")]
                redirections_counter[(item_id, redir_item_id)] += 1


            elif line.startswith("      <id>"):
                revision_id = line[len("      <id>"):-len("</id>\n")]
            elif line.startswith("      <timestamp>"):
                timestamp = line[len("      <timestamp>"):-len("</timestamp>\n")]
            elif line.startswith("      <format>"):
                format = line[len("      <format>"):-len("</format>\n")]
                format_counter[format] += 1
            elif line.startswith("      <text bytes"):
                # Quantitative Data Analysis: Examine text markup attributes and their properties
                text_xml_attribute_keys = get_xml_attribute_information(line, output="key")
                text_xml_attribute_values = get_xml_attribute_information(line, output="value")

                text_attribute_counter[text_xml_attribute_keys] += 1
                for attribute, value in zip(text_xml_attribute_keys, text_xml_attribute_values):
                    text_attribute_value_correlation[attribute][value] += 1

                if format == 'application/json':
                    # Parse Text
                    text = line[line.find('>') + 1: -len('</text>') - 1]

                    if len(text) > 0:
                        text_attribute_text_length_correlation[text_xml_attribute_keys]['len(text) > 0'] += 1
                        # Unescape text
                        text = unescape(text)
                        item_dict = json.loads(text)
                        item_dict_key_counter[tuple(item_dict.keys())] += 1
                        if 'type' in item_dict and 'id' in item_dict:
                            item_dict = WikidataItem(item_dict)

                            wikidata_compatibility_counter['compatible'] += 1

                            claims = item_dict._entity_dict['claims']
                            # # Quantitative Data Analysis: Examine text markup attributes and their properties
                            item_claim_growth[item_id].append(len(claims))

                            if len(claims) > 0:
                                claims_counter['>0'] += 1
                                text_attribute_claim_length_correlation[text_xml_attribute_keys]['len(claims) > 0'] += 1

                                # Examine Mainsnak
                                for property, claim_list in claims.items():
                                    len_claim_list_counter[len(claim_list)] += 1

                                    for claim_dict in claim_list:
                                        len_claim_dict_counter[len(claim_list)] += 1
                                        mainsnak = claim_dict['mainsnak']
                                        mainsnak_snaktype_counter[mainsnak["snaktype"]] += 1
                                        claim_rank_types_counter[claim_dict["rank"]] += 1
                                        claim_dict_keys_counter[tuple(claim_dict.keys())] += 1
                                        mainsnak_dict_key_counter[tuple(mainsnak.keys())] += 1
                                        if "datavalue" in mainsnak:
                                            mainsnak_datavalue_types_counter[mainsnak["datavalue"]["type"]] += 1
                                            if "entity-type" in mainsnak["datavalue"]["value"]:
                                                mainsnak_value_entity_types_counter[
                                                    mainsnak["datavalue"]["value"]["entity-type"]] += 1
                                                mainsnak_datavalue_types_value_entity_type_cooccurences[
                                                    mainsnak["datavalue"]["type"]][
                                                    mainsnak["datavalue"]["value"]["entity-type"]] += 1

                                                if len(mainsnak_datavalue_types_value_samples[
                                                           mainsnak["datavalue"]["value"]["entity-type"]]) < 10:
                                                    mainsnak_datavalue_types_value_samples[
                                                        mainsnak["datavalue"]["value"]["entity-type"]].append(
                                                        mainsnak["datavalue"]["value"])

                                            if type(mainsnak["datavalue"]["value"]) == dict:
                                                mainsnak_datavalue_value_keys_counter[
                                                    tuple(mainsnak["datavalue"]["value"].keys())] += 1

                                                if len(mainsnak_datavalue_value_keys_samples[
                                                           tuple(mainsnak["datavalue"]["value"].keys())][item_id]) < 5:
                                                    mainsnak_datavalue_value_keys_samples[
                                                        tuple(mainsnak["datavalue"]["value"].keys())][item_id].append(
                                                        {"item_id": item_id,
                                                         "property": property,
                                                         "mainsnak_datavalue_value": mainsnak["datavalue"]["value"]}
                                                    )

                                        # Randomly sample mainsnak content
                                        if len(mainsnak_key_content_dict[tuple(mainsnak.keys())]) < 500:
                                            random_number = randint(1, 1000)
                                            if random_number < 500:
                                                mainsnak_key_content_dict[tuple(mainsnak.keys())].append(mainsnak)

                            elif len(claims) == 0:
                                claims_counter['=0'] += 1
                                text_attribute_claim_length_correlation[text_xml_attribute_keys][
                                    'len(claims) == 0'] += 1
                        # Write gathered data into dict
                        else:
                            incompatible_items_list.append(item_dict)
                            wikidata_compatibility_counter['incompatible'] += 1

                    else:
                        text_bytes_execeptional_contents.append(line)
                        text_attribute_text_length_correlation[text_xml_attribute_keys]['len(text) == 0'] += 1

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
                is_wikidata_entity = False

        print(datetime.now().strftime("%H:%M:%S"))

        print("Quantitative data analysis results:")
        print("-- Different titles ids:")
        pprint(title_counter)

        print("-- Different formats:")
        pprint(format_counter)

        print("-- QWikidata Compatibility Counts:")
        pprint(wikidata_compatibility_counter)

        print("-- Different item dict keys:")
        pprint(item_dict_key_counter)

        print("-- QWikidata incompatible dicts :")
        pprint(incompatible_items_list)

        print("-- QWikidata claims size:")
        pprint(claims_counter)

        print("-- Mainsnak Keys:")
        pprint(mainsnak_dict_key_counter)

        print("-- text markup attribute occurences")
        pprint(text_attribute_counter)

        print("-- text attribute key and length cooccurences")
        pprint(text_attribute_text_length_correlation)

        print("-- text attribute key and claim length cooccurences")
        pprint(text_attribute_claim_length_correlation)

        print("-- Different mainsnak value entity types counter")
        pprint(mainsnak_value_entity_types_counter)

        print("-- Different claim dict keys and their frequencies. Is \'rank\' key always in item_dict?")
        pprint(claim_dict_keys_counter)

        print("-- Different claim rank types")
        pprint(claim_rank_types_counter)

        print("-- Different mainsnak datavalue types counter")
        pprint(mainsnak_datavalue_types_counter)

        print("-- Different mainsnak datavalue -> value keys counter")
        pprint(mainsnak_datavalue_value_keys_counter)

        print("-- Different mainsnak datavalue -> value keys samples")
        print(
            "-- Look for items with only numeric ids to verify that numeric id is correct. Therefore, we reconstruct the triple (item_id, property, numeric_id) to see if it was existing.")
        pprint(mainsnak_datavalue_value_keys_samples)

        print("-- Datavalue Type -> value entity type cooccurences")
        pprint(mainsnak_datavalue_types_value_entity_type_cooccurences)

        print("-- Datavalue Type -> sampled content")
        pprint(mainsnak_datavalue_types_value_samples)

        print("-- Different Lengths of claim dicts. Are there any empty claim dicts?")
        pprint(len_claim_dict_counter)

        print("-- Different Lengths of claim lists. Are there any empty claim lists?")
        pprint(len_claim_list_counter)

        print("-- Different datastructures for mainsnak --> datavalue --> value. Are there any empty claim lists?")
        pprint(len_claim_list_counter)

        print(
            "-- Different Mainsnak --> Snak Types. Compare with num of datavalues to answer wheter \'datavalue\' is always contained when snaktype = \'value\'")
        pprint(mainsnak_snaktype_counter)

        print("-- Listed redirections. Compare number with frequency of item_dict key tuple (entity, redirect)")
        pprint(redirections_counter)

        print("\n")
        pprint(
            "--> Look what information this data represent. In case Wikidata-Items are contained, look how they can be identified in order to redesign Counter to get the exact ratio of compatible/ incompatible Wikidata-Items")

        print("-- text markup attribute value cooccurences")
        pprint(text_attribute_value_correlation)

        print("-- Mainsnak Contents (randomly sampled):")
        pprint(mainsnak_key_content_dict)

        print("-- Content with len(text) == 0")
        pprint(text_bytes_execeptional_contents)


if __name__ == '__main__':
    pass
    # filename = 'wikidatawiki-20200501-pages-meta-history8.xml-p3737834p3785858.bz2'
    # analyse_xml_dump(filename)

