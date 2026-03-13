from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet_ic as wn_ic
import pandas as pd
import matplotlib.pyplot as plt


def write_wn_info_to_csv(df: pd.DataFrame, filename: str):
	df.to_csv(filename, sep="\t")


def lch(row):
	synset_i = wn.synset('.'.join((row["word1"], row["POS"].lower(), "01")))
	synset_j = wn.synset('.'.join((row["word2"], row["POS"].lower(), "01")))
	try:
		return synset_i.lch_similarity(synset_j)
	except WordNetError as e:
		return -1.0


def wup(row):
	synset_i = wn.synset('.'.join((row["word1"], row["POS"].lower(), "01")))
	synset_j = wn.synset('.'.join((row["word2"], row["POS"].lower(), "01")))
	try:
		return synset_i.wup_similarity(synset_j)
	except WordNetError as e:
		return -1.0


def jcn(row, icname="ic-bnc.dat"):
	synset_i = wn.synset('.'.join((row["word1"], row["POS"].lower(), "01")))
	synset_j = wn.synset('.'.join((row["word2"], row["POS"].lower(), "01")))
	ic = wn_ic.ic(icname)
	try:
		return synset_i.jcn_similarity(synset_j, ic)
	except WordNetError as e:
		return -1.0


if __name__ == "__main__":
	sldf = pd.read_csv("SimLex-999/SimLex-999.txt", sep="\t")
	print("SimLex-999 loaded!")
	sldf["word1word2"] = sldf["word1"] + " " + sldf["word2"]
	sldf["lch"] = sldf.apply(lch, axis=1)
	sldf["wup"] = sldf.apply(wup, axis=1)
	sldf["jcn"] = sldf.apply(jcn, axis=1)
	wndf = pd.DataFrame((sldf["word1word2"], sldf["lch"], sldf["wup"], sldf["jcn"])
		#,columns=["lch", "wup", "jcn"]
		)
	write_wn_info_to_csv(wndf, "out-simlex-similarities.csv")
