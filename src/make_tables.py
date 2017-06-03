import pickle
import numpy as np
import os
import sys
from dataset import Dataset
from lsc_subj_obj_transitive_verbs import SubjectObjectTransitiveVerbClasses
from lsc_subj_intransitive_verbs import SubjectIntransitiveVerbClasses
from lsc_verb_noun_pairs import LSCVerbClasses


data_model = pickle.load( open( "../out/all_pairs_lcs-50-100.pkl", "rb" ))
dataset  =  pickle.load( open( "../data/all_pairs-t3000.pkl", "rb" ))


def make_table_rows(dataset, data_model, cls):

	p_vc_c = data_model.p_vc[:,cls]
	p_vc_c_idx = np.argsort(p_vc_c)[::-1]

	p_nc_c = data_model.p_nc[:,cls]
	p_nc_c_idx = np.argsort(p_nc_c)[::-1]

	rows = []
	for v_idx in range(20):

		vp_idx = p_vc_c_idx[v_idx]
		vp = dataset.vps[vp_idx]

		verb_idx = dataset.vs_dict[vp[0]]
		p_idx = dataset.ps_dict[vp[1]]

		row = '\\' + " {0:.4f}".format(data_model.p_vc[verb_idx,cls]) + ' & \\verb|' + str('.'.join(vp)) + "|  "

		for n_idx in range(20):
			noun_idx = p_nc_c_idx[n_idx]
			if (vp_idx,noun_idx,  verb_idx,  p_idx) in dataset.ys_dict:
				row += '& $\\bigcdot$  '
			else:
				row += "&    "

		row += '\n'
		rows.append(row)
	return "\\\\ ".join(rows)


def make_top_rows(data_model, cls=0):

	prob = data_model.p_c[cls]

    p_nc_c = data_model.p_nc[:,cls]
	p_nc_c_idx = np.argsort(p_nc_c)[::-1]

	row1 = "\\begin{tabular}[c]{@{}l@{}}Class " + str(cls + 1) + "\\ PROB  " + str(
		round(prob, 6)) + "\end{tabular} &     "
	row2 = "&     "
	for n_idx in range(20):
		noun_idx = p_nc_c_idx[n_idx]
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
	caption = "Test Table"
	label = "my-label"


	top = table_top(caption, label)
	# print top
	top_rows = make_top_rows(data_model, cls)
	# print top_rows
	middle_rows = make_table_rows(dataset, data_model, cls)
	# print middle_rows
	bottom = table_bottom()
	# print bottom
	print(top + top_rows + middle_rows + bottom)

	with open("../out/table.txt", 'w', encoding='utf-8') as f:
		f.write(top + top_rows + middle_rows + bottom)

