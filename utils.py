from PIL import Image
import io
import os
import gdown
from zipfile import ZipFile

# https://drive.google.com/file/d/19anC-79_sNyWCzyIx2qUr7ZLOrbiBVln/view?usp=share_link

def get_model_file_gdrive( folder_path : str  = "yolov5.zip"  , folder_id : str = "19anC-79_sNyWCzyIx2qUr7ZLOrbiBVln" ):
    if not os.path.exists(folder_path):
        gdown.download(id= folder_id, output= folder_path , quiet=True )
    else:
        print("Folder {} already exists".format(folder_path))

def unzip_file( file_path : str  = "yolov5.zip" ):
    with ZipFile(file_path, 'r') as zipObj:
        zipObj.extractall(".")
    print("Unzipped file : {}".format(file_path))

if __name__ == "__main__":
    get_model_file_gdrive()
    unzip_file()