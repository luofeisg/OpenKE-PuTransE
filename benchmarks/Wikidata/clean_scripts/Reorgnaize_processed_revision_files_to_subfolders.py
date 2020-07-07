from pathlib import Path

def main():
    wikidata_path = Path.cwd().parents[0]
    print("Current Path is {}.".format(wikidata_path))
    revs_path = wikidata_path / "revision_files"
    processed_revs_path = revs_path/ "processed_revision_files"

    # Get list of processed rev markers
    revision_file_processed_markers = [rev_file for rev_file in processed_revs_path.iterdir() if rev_file.is_file()]

    # Move marker files to dump subfolder
    for f in revision_file_processed_markers:
        dump_name = f.name[f.name.find("wikidata"):f.name.find("_Q")]
        dump_subfolder = processed_revs_path / dump_name
        dump_subfolder.mkdir(exist_ok=True)
        f.rename(dump_subfolder / f.name)


if __name__ == '__main__':
    main()