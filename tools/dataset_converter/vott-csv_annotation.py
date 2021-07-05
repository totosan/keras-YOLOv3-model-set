#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class_count = {}

parser = argparse.ArgumentParser(description='convert COCO dataset annotation to txt annotation file')
parser.add_argument('--dataset_path', type=str, required=False, help='path to csv dataset, default=%(default)s', default=os.getcwd()+'/../../vott-csv-export/')
parser.add_argument('--output_path', type=str, required=False,  help='output path for generated annotation txt files, default=%(default)s', default='./')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions, default=%(default)s', default=os.path.join(os.getcwd(),'configs/custom_classes.txt'))
parser.add_argument('--has_header', required=False, help='indicates, either the annotation file has a header line or not', default=False, action="store_true" )
args = parser.parse_args()

def convertEachLine(line, classes) -> []:
    convertedLines = list()
    for line in lines:
        for key, value in classes.items():
            source = ',"'+key+'"'
            line = line.replace(source,","+str(value))
        line = line.strip().replace('"','').replace(',',' ',1)
        convertedLines.append(line)
    return convertedLines

def readSourceAnnotationFile(annotation_file):
        with open(annotation_file) as f:
            lines = f.readlines()
        if args.has_header:
            lines.pop(0)
        return lines

def getClasses(lines):
    classes = dict()
    for line in lines:
        items = line.split(',')
        classStr = items[-1].strip().replace('"','')
        number = classes.get(classStr,len(classes))
        classes[classStr] = number
    return classes

def writeNewAnnotaionFile(lines, output_path):
    with open(output_path,'w+') as f:
        for line in lines:
            f.write(line+'\n')    # write line with line-seperator
            
def writeClassesFile(classs_path: str, classes: dict):
    with open(classs_path,"w+") as f:
        for item,val in classes.items():
            newLine = '\r' if val > 0 else '' 
            f.write(newLine + item)

# get real path for dataset
dataset_realpath = os.path.realpath(args.dataset_path)

# read lines from source
lines = readSourceAnnotationFile(args.dataset_path)

# generate class list
classes = getClasses(lines)

# convert
lines = convertEachLine(lines, classes)

# write back to target
writeNewAnnotaionFile(lines, args.output_path)

# write classes file
writeClassesFile(args.classes_path,classes)