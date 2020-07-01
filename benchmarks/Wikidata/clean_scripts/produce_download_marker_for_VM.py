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
    wikidata_dump_date = "20200501"
    wikidata_path = Path.cwd()
    print("Current Path is {}.".format(wikidata_path))

    # Extract revision information about triple
    xml_dumps_path = Path.cwd() / "xml_dumps_{}".format(wikidata_dump_date)
    downloaded_marker_folder = xml_dumps_path / "downloaded_dumps"
    downloaded_marker_folder.mkdir(exist_ok=True)
    xml_dump_file_list = get_wikidata_dump_filelist()

    for f in xml_dump_file_list:
        dump_file_name = f["filename"]
        if dump_file_name == "wikidatawiki-20200501-pages-meta-history27.xml-p56826936p56827971.bz2":
            break

        else:
            downloaded_marker_file = downloaded_marker_folder / "{}.downloaded".format(dump_file_name)
            downloaded_marker_file.touch()

if __name__ == '__main__':
    main()