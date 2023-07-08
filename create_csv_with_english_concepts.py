import csv
import pandas as pd
import configparser
import json



def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


only_english = []
with open("conceptnet-assertions.csv", encoding="utf8") as f:
    for line in f.readlines():
        ls = line.split('\t')
        if ls[1].startswith('/c/en/') and ls[2].startswith('/c/en/'):
            rel = ls[0].split("/")[-1].lower()
            head = del_pos(ls[1]).split("/")[-1].lower()
            tail = del_pos(ls[2]).split("/")[-1].lower()

            if not head.replace("_", "").replace("-", "").isalpha():
                continue

            if not tail.replace("_", "").replace("-", "").isalpha():
                continue

            only_english.append("\t".join([rel, head, tail]))

with open("english_assertions.csv", "w", encoding="utf8") as f:
    f.write("\n".join(only_english))


