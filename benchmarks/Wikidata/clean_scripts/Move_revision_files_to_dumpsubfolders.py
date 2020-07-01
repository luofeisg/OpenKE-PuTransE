from pathlib import Path

def main():
    wikidata_path = Path.cwd().parents[0]
    revs_path = wikidata_path / "revision_files"
    print("Current Path is {}.".format(wikidata_path))

    # Get revision file list
    json_revision_files = [rev_file for rev_file in revs_path.glob("*.json.bz2") if
                           rev_file.is_file() and not rev_file.name.startswith("redirected")]

    # Move revision files to dump subfolder
    for f in json_revision_files:
        dump_filename = f.name[f.name.find("wikidata"):f.name.find("_Q")]
        dump_subfolder = revs_path / dump_filename
        dump_subfolder.mkdir(exist_ok=True)
        f.rename(dump_subfolder / f.name)


if __name__ == '__main__':
    main()