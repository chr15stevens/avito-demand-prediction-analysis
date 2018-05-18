import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

train = pd.read_csv('train_tmp.csv')

print(train.head())
print(train.axes)

groupableColumnLabels = ['region', 'city', 'parent_category_name', 'category_name', 'user_type']
groups = []
groupDataframes = []

# counts of groups of each column
for groupableColumnLabel in groupableColumnLabels:
    group = train.groupby([groupableColumnLabel])
    groups.append(group)
    print(groupableColumnLabel, len(group))

# give an overview of the group data if they are not too large
i = 0
for group in groups:
    print('@@@', groupableColumnLabels[i])
    if groupableColumnLabels[i] != 'city' and groupableColumnLabels[i] != 'image_top_1':
        for key, item in group:
            print(key)
        
    i += 1

# form subplot layout
fig, axes = plt.subplots(nrows=2, ncols=2)
axesArray = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

# form groups and plot data.
i = 0
for groupableColumnLabel in groupableColumnLabels:
    if groupableColumnLabel != 'city' and groupableColumnLabel != 'image_top_1':
        # i am cheating here and adding my counts into the image_top_1 column. Easiest way I could figure out how to do that.
        groupPlotDf =  train.groupby([groupableColumnLabel]).agg({'deal_probability': 'mean', 'image_top_1': 'count'}).reset_index()
        ax = groupPlotDf.plot.bar(groupableColumnLabel, 'image_top_1', ax=axesArray[i], title=groupableColumnLabel, legend=False)
        # plot the secondary deal probability line
        groupPlotDf['deal_probability'].plot(secondary_y=True, ax=ax)
        # some tidying of the graph tick labels, these dont quite work with the secondary line unfortunately.
        x_axis = ax.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)
        print(ax.axes.get_xticklabels())
        ax.axes.set_xticklabels([ticklabel.get_text()[:9] for ticklabel in ax.axes.get_xticklabels()])
        i += 1

plt.tight_layout()
plt.show()
