#! /usr/bin/bash
while true
do
    python model.py admissions-test/admissions_clean.txt admissions-test/admiss.model 5000 admissions-test/admissions-model.pth
done
