#!/bin/python3

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
from pycaret.anomaly import *

from pandas.plotting import parallel_coordinates
import plotly.express as px


# figures inline in notebook
#%matplotlib inline

np.set_printoptions(suppress=True)

DISPLAY_MAX_ROWS = 150  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

#startT=1594978929971285872

query='''
import "experimental"

pointInTime=1596171300297041697

from(bucket: "f5_v2_telegraf/autogen")
|> range(
         start: experimental.subDuration(
                  d: 30m,
                  from: time(v: pointInTime),
               ),
         stop: experimental.addDuration(
                  d: 60m,
                  to: time(v: pointInTime),
               )
   )
|> filter(fn: (r) =>
     (
      r._measurement =~ /logPemSubscribermetrics|pemflowmetrics|pemfwnatmetrics|logpemclassificationmetrics/
     )
     and
     (r._field !~ /log_message/ and r._field !~ /_msec/ and r._field !~ /errdefs_msgno/ and r._field !~ /log_severity_code/ and r._field !~ /timestamp_msec/ and r._field != "timestamp" and r._field != "severity")
   )
|> window(every: 30s)
|> mean()
|> group(columns: ["_time", "_start", "_stop"], mode: "except")
|> duplicate(column: "_stop", as: "_time")
|> pivot(
        rowKey: ["_time"],
        columnKey: ["_measurement","_field"],
        valueColumn: "_value"
   )
|> drop(columns: ["_start", "_stop"])
|> group()

'''

query_unseen='''
import "experimental"

pointInTime=1596171300297041697

from(bucket: "f5_v2_telegraf/autogen")
|> range(
         start: experimental.addDuration(
                  d: 60m,
                  to: time(v: pointInTime),
               ),
         stop: experimental.addDuration(
                  d: 90m,
                  to: time(v: pointInTime),
               )
   )
|> filter(fn: (r) =>
     (
      r._measurement =~ /logPemSubscribermetrics|pemflowmetrics|pemfwnatmetrics|logpemclassificationmetrics/
     )
     and
     (r._field !~ /log_message/ and r._field !~ /_msec/ and r._field !~ /errdefs_msgno/ and r._field !~ /log_severity_code/ and r._field !~ /timestamp_msec/ and r._field != "timestamp" and r._field != "severity")
   )
|> window(every: 30s)
|> mean()
|> group(columns: ["_time", "_start", "_stop"], mode: "except")
|> duplicate(column: "_stop", as: "_time")
|> pivot(
        rowKey: ["_time"],
        columnKey: ["_measurement","_field"],
        valueColumn: "_value"
   )
|> drop(columns: ["_start", "_stop"])
|> group()

'''

client = InfluxDBClient(url="http://localhost:8086", token="xxx", org="xxx", debug=False)
system_stats = client.query_api().query_data_frame(org="xxx", query=query)
system_stats = system_stats.infer_objects().convert_dtypes().drop(columns=["result", "table"])

dfunseen = client.query_api().query_data_frame(org="xxx", query=query_unseen)
dfunseen = dfunseen.infer_objects().convert_dtypes().drop(columns=["result", "table"])


# lets swap the first col position with _time
def swapTimeFieldToFirst(sortedIdx, columnNames):
  timeFieldIdxPos = [ i for i, ci in enumerate(sortedIdx) if columnNames[ci].find('_time') >= 0]
  timeFieldIdxPos = timeFieldIdxPos[0]
  print("{} slot have time columnName idx {} with name {}".format(timeFieldIdxPos, sortedIdx[timeFieldIdxPos], columnNames[sortedIdx[timeFieldIdxPos]]))
  zeroPosIdxValue=sortedIdx[0]
  sortedIdx[0] = sortedIdx[timeFieldIdxPos]
  sortedIdx[timeFieldIdxPos] = zeroPosIdxValue

#lets bring the '_ip' cols up in the list
def interestedCols(idx, name):
  if (name.find('_time') >= 0 or name.find('_ip') > 0 or name.find('pem_subscriber_id') >= 0):
    print(name)
    return idx
  else:
    return -idx

_cols=system_stats.columns.tolist()
sortedIdx=list(map(lambda x: x if x > 0 else -x, 
              sorted( 
                list(map(lambda x: interestedCols(x[0], x[1]), enumerate(_cols))),
                key=lambda x: x < 0)
             ))

swapTimeFieldToFirst(sortedIdx, _cols)
_cols_sorted=[_cols[i] for i in sortedIdx]
print()
print()
print("Sorted Columns {}".format(_cols_sorted))
print()
print()

system_stats = system_stats[_cols_sorted]
dfunseen = dfunseen[_cols_sorted]

dim_cols = list(system_stats.select_dtypes(include=['string']).columns)
print("Dimension Columns {}".format(dim_cols))
print()

id_cols = list(system_stats.select_dtypes(include=['datetime64', 'datetime', 'datetimetz']).columns)
print("ID Columns {}".format(id_cols))
print()

tgt_cols = list(system_stats.select_dtypes(include=['number']).columns)
print("Target Variables {}".format(tgt_cols))
print()

system_stats[dim_cols] = system_stats[dim_cols].fillna(method='bfill').fillna(method='ffill')
#system_stats[tgt_cols] = system_stats[tgt_cols].fillna(0)
system_stats[tgt_cols] = system_stats[tgt_cols]. \
                           astype('float64', copy=False). \
                           apply(pd.to_numeric, errors='coerce'). \
                           interpolate(method='linear', limit_direction='forward', axis=1)

dfunseen[dim_cols] = dfunseen[dim_cols].fillna(method='bfill').fillna(method='ffill')
#dfunseen[tgt_cols] = dfunseen[tgt_cols].fillna(0)
dfunseen[tgt_cols] = dfunseen[tgt_cols]. \
                           astype('float64', copy=False). \
                           apply(pd.to_numeric, errors='coerce'). \
                           interpolate(method='linear', limit_direction='forward', axis=1)

print()
print("Total Training Dataset Received {}".format(system_stats.shape))
print("Unseen dataset {}".format(dfunseen.shape))
print()
print()

df = system_stats.copy().drop(columns=id_cols)
#print(df[tgt_cols].head())

cormatrix = df.corr()
#display(cormatrix)

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
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    cormatrix.drop(1, axis=0)
    return cormatrix.head(numtoreport)

topcorr = mosthighlycorrelated(df, 20)
display(topcorr)

mask = np.triu(np.ones_like(cormatrix, dtype=np.bool))
cbar_kws = {"orientation":"vertical", 
            "shrink":1,
#            'extend':'min', 
#            'extendfrac':0.1, 
#            "ticks":np.arange(0,22), 
#            "drawedges":True,
           }
#f, ax = plt.subplots(figsize=(20, 10))
#heatmap = sns.heatmap(cormatrix, vmin=-1, vmax=1, square=False, center=0, annot=True,cmap="BrBG", mask=mask, cbar_kws=cbar_kws)
##heatmap = sns.heatmap(cormatrix, vmin=-1, vmax=1, square=False, center=0, annot=True,cmap="RdYlGn")
#heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), rotation=20)
#heatmap.xaxis.tick_top()
#plt.show()



anomaly = system_stats.copy()
#numeric_sts = anomaly.select_dtypes(include=['float64','int64'])
#numeric_sts = numeric_sts.apply(np.nanmean)
#print(numeric_sts)

#outliers = get_outliers(anomaly)
#print(outliers.head())

def printTopN(df, topK):
  scorecol = next((c for c in df.columns if c.find('_Score') > 0), 'Score')
#  scorecol='Score'
  print(df.nlargest(topK, scorecol))
#  perGrp=df.groupby(['dest_ip', 'source_ip']).apply(lambda x: x.nlargest(1, [scorecol])).reset_index(drop=True)
#  print(perGrp.nlargest(topK, scorecol))
  #.sort_values('Score', ascending=False)
  

topNOutliers=10

#intialize the setup
print("==============[ setup ]===================================")
exp_ano = setup(anomaly,
                numeric_imputation='mean',
                categorical_features=dim_cols,
                numeric_features=tgt_cols,
                ignore_features=['_time'],
#                normalize = True,
#                normalize_method = 'robust',
                #feature_selection = True,
#                remove_multicollinearity = True,
#                multicollinearity_threshold = 0.6,
#                pca = True, #pca_components = 10,
#                pca_method = 'kernel',
                ignore_low_variance = True,
                silent=True)
modelsList = models()
print(modelsList)
print()
print()
outlier_predictions=[]
for _id, _name in zip(modelsList.index.to_list(), list(modelsList['Name'])):
#  if _id.find('knn') >= 0 or _id.find('histo') >= 0 or _id.find('sod') >= 0 or _id.find('sos') >= 0 or _id.find('mcd') >= 0 or _id.find('cof') >= 0:
  if not _id.find('iforest') >= 0:
    continue
  print("=================================================")
  print("About to create model: {} ({}) ".format(_id, _name))
  _model = create_model(_id)
  _assigned_model = assign_model(_model, transformation=True)
  print("Topic Identifications")
  print(_assigned_model.head())
  print(_assigned_model.info())
  plot_model(_model)
  print("Predicting.......")
  _prediction = predict_model(_model, data = dfunseen)
  _outliers = _prediction[_prediction["Label"] == 1]

  print("Drop Label col and renaming Score field") 
  _outliers = _outliers.drop(columns=['Label'], axis=1)
  _outliers = _outliers.rename(columns={'Score': _id + '_Score'}) 

  print("{} determined {} ".format(_id, _outliers.shape))
#  print(_outliers.info())
# !! this corrupts the pandas DF and subsequent index lookup (df.nlargest) fails !!
#  _outliers.to_csv(_id + "_outliers.csv")
  outlier_predictions.append( [(_id, _name), (_outliers, _prediction)] )

print()
print()
print("PRINTING top N Outliers now ---------------------")
print()
for entry in outlier_predictions:
  _id = entry[0][0]
  _name = entry[0][1]
  _outliers = entry[1][0]
  _predictions = entry[1][1]
  print("{} model {}".format(_id, _name))
  print("---------------------------------------------------")
  if _outliers.shape[0] == 0:
    print("No Outliers FOUND ! {}".format(_outliers.shape))
    print()
    print("---------------------------------------------------")
    continue
  printTopN(_outliers, topNOutliers)
  print(_outliers.shape)
  print()
  print("---------------------------------------------------")
  _combinecols=[
                 #['dest_ip', 'pem_subscriber_id', 'source_ip'],
                 ['dest_ip', 'source_ip'],
                 ['subscriberidtype', 'subscribername', 'subscribertype'],
                 ['translated_dest_ip', 'translated_source_ip'],
                 ['towername', 'device_product', 'device_vendor', 'devicename', 'deviceos'],
               ]

  _dropcols = ['Label', 'Score', 'appname', 'ip_protocol', 'hostname', 'context_name', 'dest_port', 'device_version', 'errdefs_msg_name', 'event_name', 'facility', 'host', 'log_hostname', 'log_severity', 'route_domain', 'severity', 'source_port', 'translated_dest_port', 'translated_route_domain', 'translated_source_port', 'calledname', 'callingname',  'entity', 'iplist', 'origin', 'slotid', 'username', 'applicationname', 'categoryname', 'logPemSubscribermetrics_aggrinterval', 'logpemclassificationmetrics_aggrinterval', 'pemfwnatmetrics_duration', 'logpemclassificationmetrics_eoctimestamp', 'logPemSubscribermetrics_eoctimestamp', 'logpemclassificationmetrics_bytesin', 'logpemclassificationmetrics_bytesout', 'logpemclassificationmetrics_hitcount']

  _toPlot = _predictions
  for _cols in _combinecols:
    newColName = '-'.join([i for i in _cols])
    _toPlot[newColName] = _toPlot[ _cols ].astype(str).apply('_'.join, axis=1)
    _toPlot = _toPlot.drop(columns=_cols, axis=1)

  _toPlot = _toPlot.drop(_dropcols, axis=1)
  _toPlot = _toPlot.convert_dtypes()
  renamePairs = [ (c, c[len('logPemSubscribermetrics_'):]) for c in _toPlot.columns.to_list() if c.find('logPemSubscribermetrics_') >= 0]
  oldCols = [ c[0] for c in renamePairs ]
  newCols = [ c[1] for c in renamePairs ]
  _toPlot = _toPlot.rename(columns=dict(zip(oldCols,newCols)))
  _toPlot = _toPlot.set_index(_toPlot._time)
  print(_toPlot.shape)
  print(_toPlot.info())

  #for _metric in tgt_cols:
  #n_tcols = [ c for c in tgt_cols if c.find(_metric) < 0 ]
  # n_plot = _toPlot.copy().drop(columns=n_tcols, axis=1)
  x_cols = list(_toPlot.select_dtypes(include=['string']).columns)
  y_cols = list(_toPlot.select_dtypes(include=['number']).columns)

  x_cols = [x for x in x_cols if x.find('pem_subscriber_id') < 0 and x.find('subscriberidtype-') < 0 and x.find('towername-') < 0]
  y_cols = [y for y in y_cols if y.find('eoctimestamp') < 0 and y.find('aggrinterval') < 0]
  print(x_cols)
  print(y_cols)
  g = sns.pairplot(_toPlot,
               x_vars=x_cols,
               y_vars=y_cols,
#               hue='dest_ip-pem_subscriber_id-source_ip', 
               hue='pem_subscriber_id', 
               diag_kind="kde", height=2.5,
               palette = 'hls')


#  g = sns.PairGrid(_toPlot,
#                   y_vars=y_cols,
#                   x_vars='_time',
#                   hue='pem_subscriber_id', 
#                   height=4)
#  g.map(plt.scatter)
##  g.legend(loc='upper left')
##  g.set_xticklabels(rotation=30)
##  g.set_yticklabels(rotation=)
#  plt.xticks(rotation=30)
#  plt.yticks(rotation=30)
#
#  g.despine(left=True)

  plt.show()

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#fig = make_subplots(rows=2, cols=2, start_cell="bottom-left")
  
#fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
#              row=1, col=1)
#  
#  f = _toPlot[ y_cols + ['pem_subscriber_id'] ]
#  f = f.dropna()
#  parallel_coordinates(f, 'pem_subscriber_id')

  #grid = grid.map_upper(plt.scatter)
  #grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
  #                   edgecolor = 'k')# Map a density plot to the lower triangle
  #grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')


# creating a model
#iforest = create_model('iforest')
#print(iforest)
#iforest_outliers = assign_model(iforest)
#iforest_predict  = predict_model(iforest, data = anomaly)
#iforest_anomalous_rows = iforest_predict[iforest_predict["Label"] == 1]
#print(iforest_anomalous_rows.shape)
#
### plotting a model
##plot_model(iforest, plot = 'umap')
#
#
### creating a model
#knn = create_model('knn')
### plotting a model
##plot_model(knn)
#
#knn_outliers=assign_model(knn)
## generate predictions using trained model
#knn_predictions = predict_model(knn, data = anomaly)
#knn_predictions_rows = knn_predictions[knn_predictions["Label"] == 1]
#printTopN(knn_predictions_rows, topNOutliers)
#print(knn_predictions_rows.shape)
#

client.__del__()

