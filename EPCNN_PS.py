import argparse as ap
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from ete3 import NCBITaxa



def read_params(args):
    parser = ap.ArgumentParser(description='Specify the probability')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')
    return vars(parser.parse_args())


def read_files(file_name):


    # file_name='Karlsson_T2D'

    known = pd.read_csv("data/" + file_name+'_known.csv', index_col=0)
    unknown = pd.read_csv("data/" + file_name+'_unknown.csv', index_col=0)

    y = pd.read_csv("data/" + file_name+'_y.csv', index_col=0)
    le = preprocessing.LabelEncoder()

    y=np.array(y).ravel()
    y = le.fit_transform(y)
    return known, unknown, y




# known=pd.read_csv('Zeller_CRC_known.csv',index_col=0)
# unknown=pd.read_csv('Zeller_CRC_unknown.csv',index_col=0)

par = read_params(sys.argv)

file_name = str(par['fn'])



known, unknown,y=read_files(file_name)

# since we got taxaid from the MicroPro, therefore after using PhyloT,
# some taxid is will be not accurate because some of them are updated,
# so we need to replace some of them.

# for KT2D
known=known.rename(columns={'330':'301','697046':'645','758602':'1073996',
                        '1315956':'2496551','1834200':'1796646','1870930':'1812935'})

# for QT2D
# known=known.rename(columns={'330':'301','1834200':'1796646'})

# for QLC
# known=known.rename(columns={'330':'301','1834200':'1796646'})

# for ZCRC
# known=known.rename(columns={'330':'301','1834200':'1796646',
#                             '319938':'288004',
#                             '1166016':'1905730'})



# here as we descriped in the paper, we PhyloT to generate the tree,
# since PhyloT is not free, so here we offer a free way to genetate by using ETE3


raw_id=known.columns.values.tolist()
ncbi = NCBITaxa()

# Also, we can use the Newick obtained file to get the tree by using PhyloT, just like the
# description in our paper


# import ete3
# tree=ete3.Tree("tree.txt",format=8)
# print(tree)


tree = ncbi.get_topology(raw_id)
print (tree.get_ascii(attributes=["taxid"]))



order = []
num = 1
for node in tree.traverse(strategy='levelorder'):
    if node.is_leaf():
        order.append(node.name)

postorder = []
num = 1
for node in tree.traverse(strategy='postorder'):
    if node.is_leaf():
        postorder.append(node.name)

temp = []
for i in order:
    if i in known.columns:
        temp.append(i)

order = temp


temp1 = []
for i in postorder:
    if i in known.columns:
        temp1.append(i)

postorder  = temp1


known_Xl=known[order]
known_Xp=known[postorder]



known_Xl.to_csv(file_name+'_knownl.csv')
known_Xp.to_csv(file_name+'_knownp.csv')

# for unknown features, we just arrange the taxa with at least genus levels.

import xlrd
data = xlrd.open_workbook("data/unknown_name.xlsx")

# for the first dataset, therefore the sheet number is 0,1,2,3 respectively
table = data.sheets()[0]


binname =table.col_values(0)[2:-1]
binname=["V"+str(int(i)) for i in binname ]
unknown_structure = unknown[binname]
unbinname=[]
unknown_id=unknown.columns.values.tolist()
for i in unknown_id:
    if i not in unknown_structure:
        unbinname.append(i)
unknown_nostructure=unknown[unbinname]


structure_taxaid=table.col_values(2)[2:-1]
structure_taxaid=[str(int(i)) for i in structure_taxaid]
unknown_structure.columns = structure_taxaid

ncbi = NCBITaxa()



tree = ncbi.get_topology(structure_taxaid)

order = []
num = 1
for node in tree.traverse(strategy='levelorder'):
    if node.is_leaf():
        order.append(node.name)

postorder = []
num = 1
for node in tree.traverse(strategy='postorder'):
    if node.is_leaf():
        postorder.append(node.name)

unknown_order = pd.concat([unknown_structure[order], unknown_nostructure], axis=1)
unknown_postorder = pd.concat([unknown_structure[postorder], unknown_nostructure], axis=1)

unknown_order.to_csv(file_name+'_unknownl.csv')
unknown_postorder.to_csv(file_name+'_unknownp.csv')


























