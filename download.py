import zipfile
import os
import ssl
import wget


def download_file(file_path, link, local_path, raw_data):

    if not os.path.exists(raw_data):

        print("No raw data folder found")

        if not os.path.exists(file_path):
            print("No zip file found")
            print("Downloading from %s, this may take a while..." % link)
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(link, file_path)
            print("\n")
            print("File Downloaded")

        else:
            print("Zip file Already Exists")

        print("Unzip File")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(local_path)

    else:
        print("Raw Data Folder already exists")

    if os.path.exists(file_path):
        os.remove(file_path)

    return raw_data