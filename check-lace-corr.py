from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML



# figures inline in notebook
#%matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)


#startT=1591664810000000000
#endT=startT+(60 * 60 * 1000000000)

#query= '''
#from(bucket:"f5_aws_telegraf/autogen")
#|> range(start:time(v:1591983301000000000), stop: time(v: (20 * 60 * 60 * 1000000) + 1591983301000000000))
#|> filter(fn: (r) =>
#    r._measurement == "docker_container_net")
#|> drop(columns:["_start", "_stop", "result", "table", "JAVA_HOME"])
#|> pivot(
#        rowKey:["_time"],
#        columnKey: ["_field"],
#        valueColumn: "_value"
#      )
#'''

query='''
import "experimental"

from(bucket:"f5_aws_telegraf/autogen")
|> range(start:time(v:1591024100000000000), 
         stop: experimental.addDuration(
		  d: 15d,
		  to: time(v: 1591024100000000000),
		)
        )
|> filter(fn: (r) =>
    r._measurement == "disk" or r._measurement == "diskio" )
|> limit(n:100)
|> pivot(
        rowKey:["_time"],
        columnKey: ["_field"],
        valueColumn: "_value"
      )
|> window(every: 1s)
|> drop(columns:["_start", "_stop", "result", "table", "JAVA_HOME"])
'''
client = InfluxDBClient(url="http://localhost:8086", token="xxx", org="xxx", debug=False)
#system_stats = client.query(query)
system_stats = client.query_api().query_data_frame(org="xxx", query=query)


print(len(system_stats))

rs=iter(system_stats)
df=next(rs)
df=df.infer_objects()
for _frame in rs:
  shape = _frame.shape
#  print("Size = {} ".format(shape))
  _frame=_frame.infer_objects()
#  print(_frame.dtypes)
  df=pd.merge(df, _frame, on='_time', how='outer')
  
df = df.select_dtypes(exclude=['object']).drop(['table_x', 'table_y', '_time'], axis = 1) 
#df = df.select_dtypes(exclude=['object']).drop(['table_x', 'table_y','_time', 'reads', 'read_time', 'read_bytes', 'total', 'inodes_total', 'merged_reads', 'write_time', 'used_percent'], axis = 1) 
print(df.info())
print(df.head())
#pd.plotting.scatter_matrix(df.loc[::, ~df.columns.isin(['_time', 'reads', 'read_time', 'read_bytes', 'total', 'merged_reads', 'write_time', 'used_percent'])], diagonal="kde", alpha=0.4)
#plt.tight_layout()
#plt.show()
#corr_matrix = np.corrcoef(df.to_numpy()).round(decimals=2)
#fig, ax = plt.subplots()
#im = ax.imshow(corr_matrix)
#im.set_clim(-1, 1)
#ax.grid(False)
#ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
#ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
#ax.set_ylim(2.5, -0.5)
#cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
#plt.show()


# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()

def mosthighlycorrelated(_df, numtoreport):
    # find the correlations
    #_df = _df.drop(['_time'], axis = 1) 
    #print(_df.info())
    cormatrix = _df.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
#    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    cormatrix.drop(1, axis=0)
    return cormatrix.head(numtoreport)

def plot_dates_values(data):
    dates = data["timestamp"].to_list()
    values = data["value"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates, values)
    plt.show()


#hinton(corr_matrix)
topcorr = mosthighlycorrelated(df, 10)
display(topcorr)
#cormatrix = np.corrcoef(df.to_numpy()).round(decimals=2)
cormatrix = df.corr()
display(cormatrix)
plt.figure(figsize=(20,20))
sns.heatmap(cormatrix, vmax=1., square=False, annot=True,cmap="RdYlGn").xaxis.tick_top()
plt.show()

#for di in range(len(system_stats)):
##  print(df.info())
##  print(df[['table', '_measurement', '_field']])
#  if( di == 0 ):
#    continue
#  df = system_stats[di]
#  shape = df.shape
#  print("Size = {} ".format(shape))
#  print(df.infer_objects().dtypes)
#  dfI = df.infer_objects()
#  corrmat = dfI.corr()
#  #display(corrmat)
#  sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()
#  plt.show()
#  print("----------------------------------------------")
#

# write DF to influx
#from influxdb_client import InfluxDBClient, Point, WriteOptions
#from influxdb_client.client.write_api import SYNCHRONOUS
## Preparing Dataframe: 
#system_stats.drop(columns=['result', 'table','start','stop'])
## DataFrame must have the timestamp column as an index for the client. 
#system_stats.set_index("_time")
#_write_client.write(bucket.name, record=system_stats, data_frame_measurement_name='cpu',
#                    data_frame_tag_columns=['cpu'])
#_write_client.__del__()

client.__del__()

