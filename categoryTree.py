#%%
from treelib import Node, Tree

  

#%%
# class name(object):
#     names= []
#     def __init__(self, names):
#         self.names = names

#%%

#cuisine tree
cu = Tree()
cu.create_node('cuisine', 'cuisine')
cu.create_node('eastern', 'eastern', parent='cuisine')
cu.create_node('western', 'western',  parent='cuisine')

cu.create_node('middle eastern', 'middle eastern',  parent='eastern')
cu.create_node('asian', 'asian',  parent='eastern')
cu.create_node('african', 'african',  parent='eastern')

cu.create_node('european', 'european', parent= 'western')
cu.create_node('american', 'american',  parent='western')
cu.create_node('mexican', 'mexican',  parent='western')

cu.create_node('desi', 'desi',  parent='asian')
cu.create_node('afghani', 'afghani',  parent='asian')
cu.create_node('east asian', 'east asian',  parent='asian')
cu.create_node('turkish', 'turkish',  parent='asian')

cu.create_node('chinese', 'chinese',  parent='east asian')
cu.create_node('thai', 'thai',  parent='east asian')

cu.create_node('italian', 'italian',  parent='european')
cu.create_node('french', 'french',  parent='european')




#%%
#display cuisine tree
#cu.show()

#%%
