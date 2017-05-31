import pickle
import numpy as np
import os
import sys
from dataset import Dataset
from lsc_subj_obj_transitive_verbs import SubjectObjectTransitiveVerbClasses
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
from lsc_verb_noun_pairs import LSCVerbClasses


data_model = pickle.load( open( "../out/all_pairs_lcs-75-150.pkl", "rb" ))
dataset  =  pickle.load( open( "../data/all_pairs-t3000.pkl", "rb" ))

p_vc_sorted = np.argsort(data_model.p_vc, axis=0)[-20:,:]
p_nc_sorted = np.argsort(data_model.p_nc, axis=0)[-20:,:]


def make_table_rows(dataset, p_vc_sorted, p_nc_sorted, cls):
	rows = []
	for v_idx in range(19, -1, -1):
		vp_idx = p_vc_sorted[v_idx, cls]
		vp = dataset.vps[vp_idx]

		verb_idx = dataset.vs_dict[vp[0]]
		p_idx = dataset.ps_dict[vp[1]]


		row = '\\' + " {0:.4f}".format(round(data_model.p_vc[verb_idx, cls], 6)) + ' & \\verb|' + str('.'.join(vp)) + "|  "

		for n_idx in range(19, -1, -1):
			noun_idx = p_nc_sorted[n_idx, cls]
			if (vp_idx,noun_idx,  verb_idx,  p_idx) in dataset.ys_dict:
				row += '& $\\bigcdot$  '
			else:
				row += "&    "

		rows.append(row)
	return "\\\\ ".join(rows)


def make_top_rows(p_nc_sorted, data_model, cls=0, prob=0):
	row1 = "\\begin{tabular}[c]{@{}l@{}}Class " + str(cls + 1) + "\\ PROB  " + str(
		round(prob, 6)) + "\end{tabular} &     "
	row2 = "&     "
	for n_idx in range(19, -1, -1):
		noun_idx = p_nc_sorted[n_idx, cls]
		row1 += "&       \\rotatebox[origin=c]{90}{      " + str(
			round(data_model.p_nc[noun_idx, cls], 6)) + "    }     "
		row2 += "&     \\rotatebox[origin=c]{90}{    " + dataset.ns[noun_idx] + "      }     "
	row1 += " \\\\" + " \\hline"
	row2 += " \\\\" + " \\hline"
	return row1 + "\n" + row2


def table_bottom():
	return "\\\\ \n" + "\hline \n" + "\end{tabular} \n" + "\end{adjustbox} \n" + "\end{table} \n"


def table_top(caption="My caption", label="my-label"):
	return "\\begin{table}[] \n" + "\centering  \n" + "\caption{" + caption + "} \n" + "\label{" + label + "} \n"  \




for cls in range(0, data_model.n_cs):
	caption = "My caption"
	label = "my-label"

	top = table_top(caption, label)
	# print top
	top_rows = make_top_rows(p_nc_sorted, data_model, cls, data_model.p_c[cls])
	# print top_rows
	middle_rows = make_table_rows(dataset, p_vc_sorted, p_nc_sorted, cls)
	# print middle_rows
	bottom = table_bottom()
	# print bottom
	print(top + top_rows + middle_rows + bottom)