from pathlib import Path
import re


def rmtree(path):
    path = Path(path)  # allow path to be a string
    assert path.is_dir()  # make sure it`s a folder
    for p in reversed(list(path.glob('**/*'))):  # iterate contents from leaves to root
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()


def main():
    wikidata_path = Path.cwd().parents[0]
    revision_files_path = wikidata_path / "revision_files"
    revision_folder_pattern = re.compile(r'[\s\S]*pages-meta-history.*\.bz2$$')
    json_revision_folder = [rev_folder for rev_folder in revision_files_path.iterdir() if rev_folder.is_dir()
                            and revision_folder_pattern.match(rev_folder.name)]
    # Move marker files to dump subfolder
    for subfolder in json_revision_folder:
        for folder_content in subfolder.iterdir():
            if folder_content.name == "buggy_revision_files":
                print("deleted folder buggy_revision_files in folder {}.".format(subfolder.name))
                rmtree(folder_content)
                Path.rmdir(folder_content)


if __name__ == '__main__':
    main()
