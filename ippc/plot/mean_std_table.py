import pandas as pd
import numpy as np

print(pd.__version__)
# Load the data

path_all_rpd = "./plot/data_source/all_rpd.pickle"
path_all_solved = "./plot/data_source/all_solved.pickle"
path_all_collection = "./plot/data_source/all_collection.pickle"

df_all_rpd = pd.read_pickle(path_all_rpd)
df_all_solved = pd.read_pickle(path_all_solved)
df_all_collection = pd.read_pickle(path_all_collection)

# df_all_rpd = df_all_rpd.drop(
#     index=['20240306_094320_rot_aug', '20240306_094346_rot_aug_batch', 'ens_reg_100_3', 'ens_reg_100_4',
#            'ens_reg_100_5'])
# df_all_solved = df_all_solved.drop(
#     index=['20240306_094320_rot_aug', '20240306_094346_rot_aug_batch', 'ens_reg_100_3', 'ens_reg_100_4',
#            'ens_reg_100_5'])
# df_all_collection = df_all_collection.drop(
#     index=['20240306_094320_rot_aug', '20240306_094346_rot_aug_batch', 'ens_reg_100_3', 'ens_reg_100_4',
#            'ens_reg_100_5'])

new_column_dict = {}
for column_name in df_all_rpd.columns:
    new_column_name = ""
    new_new_column_name_split = column_name.split("_")
    if new_new_column_name_split[0][:3] == "map":
        new_column_name = "ood"
    elif "rot" in new_new_column_name_split:
        new_column_name = 'id_rot'
    else:
        new_column_name = "id"
    new_column_dict[column_name] = new_column_name

new_index_dict = {}

for index_name in df_all_rpd.index:
    index_name_split = index_name.split("_")
    if index_name_split[0] == "augmented":
        new_index_dict[index_name] = "augmented"
    if index_name_split[0] == "baseline":
        new_index_dict[index_name] = "baseline"
    if index_name_split[0] == "ens":
        if index_name_split[2] == "100":
            new_index_dict[index_name] = "ens_reg_100"
        elif index_name_split[2] == "50":
            new_index_dict[index_name] = "ens_reg_50"
    if index_name_split[0] == "ensemble":
        new_index_dict[index_name] = "ensemble"
    if index_name_split[0] == "regularized":
        new_index_dict[index_name] = "regularized"

agent_name_list = ["baseline", "augmented", "ensemble", "regularized", "ens_reg_50", "ens_reg_100"]

df_all_rpd = df_all_rpd.rename(columns=new_column_dict)  # rename map to id, id_rot, ood
df_all_rpd = df_all_rpd.transpose()
df_all_rpd = df_all_rpd.groupby(level=0).mean()  # per agent mean for each group of maps
df_all_rpd = df_all_rpd.transpose()
df_all_rpd = df_all_rpd.rename(index=new_index_dict)  # rename agent to prepare grouping

df_all_rpd_std = df_all_rpd.groupby(level=0).std()  # std for agent based on per agent mean
df_all_rpd_mean = df_all_rpd.groupby(level=0).mean()  # mean for agent based on per agent mean

df_all_rpd_std = df_all_rpd_std.transpose()  # transpose to make it easier to reindex
df_all_rpd_mean = df_all_rpd_mean.transpose()  
df_all_rpd_mean = df_all_rpd_mean.reindex(agent_name_list, axis=1)  # reindex to make sure the order is correct
df_all_rpd_std = df_all_rpd_std.reindex(agent_name_list, axis=1)


print("================RPD=================")
print(df_all_rpd_mean)
print(df_all_rpd_std)
print("")



df_all_solved = df_all_solved.rename(columns=new_column_dict)  # rename map to id, id_rot, ood
df_all_solved = df_all_solved.transpose()
df_all_solved = df_all_solved.groupby(level=0).mean()  # per agent mean for each group of maps
df_all_solved = df_all_solved.transpose()
df_all_solved = df_all_solved.rename(index=new_index_dict)  # rename agent to prepare grouping

df_all_solved_std = df_all_solved.groupby(level=0).std()  # std for agent based on per agent mean
df_all_solved_mean = df_all_solved.groupby(level=0).mean()  # mean for agent based on per agent mean

df_all_solved_std = df_all_solved_std.transpose()  # transpose to make it easier to reindex
df_all_solved_mean = df_all_solved_mean.transpose()  
df_all_solved_mean = df_all_solved_mean.reindex(agent_name_list, axis=1)  # reindex to make sure the order is correct
df_all_solved_std = df_all_solved_std.reindex(agent_name_list, axis=1)


print("================Solved=================")
print(df_all_solved_mean)
print(df_all_solved_std)
print("")


df_all_collection = df_all_collection.rename(columns=new_column_dict)  # rename map to id, id_rot, ood
df_all_collection = df_all_collection.transpose()
df_all_collection = df_all_collection.groupby(level=0).mean()  # per agent mean for each group of maps
df_all_collection = df_all_collection.transpose()
df_all_collection = df_all_collection.rename(index=new_index_dict)  # rename agent to prepare grouping

df_all_collection_std = df_all_collection.groupby(level=0).std()  # std for agent based on per agent mean
df_all_collection_mean = df_all_collection.groupby(level=0).mean()  # mean for agent based on per agent mean

df_all_collection_std = df_all_collection_std.transpose()  # transpose to make it easier to reindex
df_all_collection_mean = df_all_collection_mean.transpose()  
df_all_collection_mean = df_all_collection_mean.reindex(agent_name_list, axis=1)  # reindex to make sure the order is correct
df_all_collection_std = df_all_collection_std.reindex(agent_name_list, axis=1)

print("================Collection=================")
print(df_all_collection_mean)
print(df_all_collection_std)
print("")

for i, index in enumerate(df_all_rpd_mean.index):
    print(f"{index}")
    print(f"& solved"
          f"&\multicolumn{2}{{c|}}{{${{{df_all_solved_mean.iloc[i, 0]:.2f}}}^{{{df_all_solved_std.iloc[i, 0]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_solved_mean.iloc[i, 1]:.2f}}}^{{{df_all_solved_std.iloc[i, 1]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_solved_mean.iloc[i, 2]:.2f}}}^{{{df_all_solved_std.iloc[i, 2]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_solved_mean.iloc[i, 3]:.2f}}}^{{{df_all_solved_std.iloc[i, 3]:.2f}}}$}} "
          f"&\multicolumn{2}{{c|}}{{${{{df_all_solved_mean.iloc[i, 4]:.2f}}}^{{{df_all_solved_std.iloc[i, 4]:.2f}}}$}} "
          f"& \multicolumn{2}{{c}}{{${{{df_all_solved_mean.iloc[i, 5]:.2f}}}^{{{df_all_solved_std.iloc[i, 5]:.2f}}}$}}\\\\\cline{{2-14}}")
    print(f"& RPD"
          f"&\multicolumn{2}{{c|}}{{${{{df_all_rpd_mean.iloc[i, 0]:.2f}}}^{{{df_all_rpd_std.iloc[i, 0]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_rpd_mean.iloc[i, 1]:.2f}}}^{{{df_all_rpd_std.iloc[i, 1]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_rpd_mean.iloc[i, 2]:.2f}}}^{{{df_all_rpd_std.iloc[i, 2]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_rpd_mean.iloc[i, 3]:.2f}}}^{{{df_all_rpd_std.iloc[i, 3]:.2f}}}$}} "
          f"&\multicolumn{2}{{c|}}{{${{{df_all_rpd_mean.iloc[i, 4]:.2f}}}^{{{df_all_rpd_std.iloc[i, 4]:.2f}}}$}} "
          f"& \multicolumn{2}{{c}}{{${{{df_all_rpd_mean.iloc[i, 5]:.2f}}}^{{{df_all_rpd_std.iloc[i, 5]:.2f}}}$}} \\\\\cline{{2-14}}")
    print(f"& collections"
          f"&\multicolumn{2}{{c|}}{{${{{df_all_collection_mean.iloc[i, 0]:.2f}}}^{{{df_all_collection_std.iloc[i, 0]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_collection_mean.iloc[i, 1]:.2f}}}^{{{df_all_collection_std.iloc[i, 1]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_collection_mean.iloc[i, 2]:.2f}}}^{{{df_all_collection_std.iloc[i, 2]:.2f}}}$}} "
          f"& \multicolumn{2}{{c|}}{{${{{df_all_collection_mean.iloc[i, 3]:.2f}}}^{{{df_all_collection_std.iloc[i, 3]:.2f}}}$}} "
          f"&\multicolumn{2}{{c|}}{{${{{df_all_collection_mean.iloc[i, 4]:.2f}}}^{{{df_all_collection_std.iloc[i, 4]:.2f}}}$}} "
          f"& \multicolumn{2}{{c}}{{${{{df_all_collection_mean.iloc[i, 5]:.2f}}}^{{{df_all_collection_std.iloc[i, 5]:.2f}}}$}}")
