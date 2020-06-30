from pathlib import Path
import re
import sys
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

def get_wikidata_dump_filelist(wikidata_dump_date=20200501):
    # Get list of xml history files and their URLs. #
    # Download and process checksums in order to    #
    # validate the downloaded files                 #

    wikidata_url = 'https://dumps.wikimedia.org'

    response = urlopen(wikidata_url + '/wikidatawiki/' + str(wikidata_dump_date))
    soup = BeautifulSoup(response, "html.parser")

    # -- Get relevant filenames and URLs of relevant dumps
    link_elements = soup.find_all('a', href=re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$'))
    dumps = [{"filename": element.getText(),
              "url": element.get('href'),
              "dumpdate": wikidata_dump_date}
             for element in link_elements]

    return dumps

def main():
    # get all dumps
    wikidata_dump_date = "20200501"
    wikidata_path = Path.cwd()
    print("Current Path is {}.".format(wikidata_path))

    # Extract revision information about triple
    xml_dump_file_list = get_wikidata_dump_filelist()
    xml_dump_file_list = [f["filename"] for f in xml_dump_file_list]

    # get file names of processed dumps
    processed_dumps = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date) / "processed_dumps"
    processed_dumps_file_list = [f.name[:f.name.find(".processed")] for f in processed_dumps.iterdir()]

    # obtain filenames which have not been processed (all dumps - processed dumps)
    unprocessed_file_list = [x for x in xml_dump_file_list if x not in processed_dumps_file_list]

    # Iterate through revision files and delete corresponding files
    revision_files_path = wikidata_path / "revision_files"
    for filename in unprocessed_file_list:
        for revision_file in revision_files_path.glob("*{}*".format(filename)):
            revision_file.unlink()
            print("Deleted {}.".format(revision_file.name))

if __name__ == '__main__':
    main()