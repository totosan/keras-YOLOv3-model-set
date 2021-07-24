import os,sys
import argparse
from pathlib import Path
from PIL import Image

def get_dataset(annotation_file, root_path=None, hasHeader=False, shuffle=True):
    if not root_path:
        root_path = "."    
    annotationFilePath = os.path.normpath(os.path.join(root_path,annotation_file))
    print(f"Annotationfile's path: {annotationFilePath}")
    with open(annotationFilePath) as f:
        lines = f.readlines()
        if hasHeader:
            lines.pop(0)
        tempLines = []
        for line in lines:
            tempLines.append(os.path.normpath(os.path.join(root_path, line.strip())))
        lines = tempLines

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)
        #np.random.seed(None)

    return lines

def main(args):
    dir = Path(args.path)
    for child in dir.iterdir():
        print(f"file: {child}")
    print("#############")
    dataset = get_dataset(annotation_file=args.csv_file,root_path=args.path,hasHeader=True,shuffle=False)
    for record in dataset:
        imgPath =str(record.split(' ')[0]).replace('%','%25')
        print(f"image name: {imgPath}")
        img = Image.open(imgPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, default=None, help = "Name of csv file")
    parser.add_argument('--path', type=str, required=True, default=None, help = "Path to the image file")
    
    args = parser.parse_args()
    main(args)