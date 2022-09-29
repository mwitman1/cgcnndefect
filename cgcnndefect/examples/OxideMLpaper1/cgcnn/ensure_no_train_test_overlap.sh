#! /bin/bash

for f in $(find . -type f -name "id_prop.csv.hold*"); do suffix=${f#*id_prop.csv.hold}; grep -Ff $f "id_prop.csv.train"$suffix; done
