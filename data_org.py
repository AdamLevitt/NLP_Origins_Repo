# Module imports
import numpy as np
import download
import os
import glob2

filename = "data.zip"
foldername = "data"
url = "https://download.pytorch.org/tutorial/" + filename

local = os.getcwd() + "/"

zip_path = local + filename
raw_path = local + foldername

download.download_file(zip_path, url, local, raw_path)
print(f'Folder "{foldername}" found at: {raw_path}')