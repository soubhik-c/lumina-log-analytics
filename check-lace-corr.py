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
|> limit(n:1000)
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
  
df = df.select_dtypes(exclude=['object']).drop(['table_x', 'table_y','reads', 'read_time', 'read_bytes', 'total', 'inodes_total', 'merged_reads', 'write_time', 'used_percent'], axis = 1) 
print(df.info())
print(df.head())
pd.plotting.scatter_matrix(df.loc[::, ~df.columns.isin(['_time', 'reads', 'read_time', 'read_bytes', 'total', 'merged_reads', 'write_time', 'used_percent'])], diagonal="kde", alpha=0.4)
plt.tight_layout()
plt.show()
corrmat = df.corr()
display(corrmat)
sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()
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

