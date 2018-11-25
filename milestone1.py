import csv
import os
from glob import glob

def main():
    with open('label.csv', 'w') as csvfile:
        fieldnames = ['guid/image', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        path = glob('deploy/test/*')
        for p in path:
            # print(p)
            # folder = p.split('/')[2]
            sub_path = glob(p + '/*_image.jpg')
            for p1 in sub_path:
                folder = p1.split('/')[2]
                id = p1.split('/')[3][:4]
                new_path = folder + '/' + id
                # print(folder, id)
                writer.writerow({'guid/image': new_path, 'label': 1})

if __name__ == '__main__':
    main()
