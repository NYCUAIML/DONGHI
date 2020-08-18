import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("0714train.csv")
tdata = pd.read_csv("0728test.csv")

data["x+"]=0
data["x-"]=0
data["y+"]=0
data["y-"]=0
tdata["x+"]=0
tdata["x-"]=0
tdata["y+"]=0
tdata["y-"]=0


#補缺失值#string補xy都沒移動#數值補平均值
def add_null(idata,colname):
    for i,j in enumerate(idata.isnull().any()):
        if j:
            if idata[colname[i]].dtype == "object":
                for x,z in enumerate(idata[colname[i]]):
                    if type(z) == float:
                        idata.iloc[x,i]="N;0;N;0"
            else:
                for x,z in enumerate(idata[colname[i]]):
                    if np.isnan(z):
                        idata.iloc[x,i]=np.mean(idata[colname[i]])
    return idata

cols = list(data.columns)
data = add_null(data,cols)

cols = list(tdata.columns)
tdata = add_null(tdata,cols)


#提取string(c15-c38,c63-c82)資料
takelist = []
for i in range(15,39):
    takelist.append("Input_C_0"+str(i))
for i in range(63,83):
    takelist.append("Input_C_0"+str(i))

def xychange(inp,tlist):
    for i in inp:
        if i in tlist:
            for z,j in enumerate(inp[i]):
                if type(j) == float:
                    continue
                j = j.split(";")
                p = 0
                while p < 4:
                    if j[p] == "R":
                        inp["x+"][z] = inp["x+"][z]+float(j[p+1])
                    if j[p] == "L":
                        inp["x-"][z] = inp["x-"][z]+float(j[p+1])
                    if j[p] == "U":
                        inp["y+"][z] = inp["y+"][z]+float(j[p+1])
                    if j[p] == "D":
                        inp["y-"][z] = inp["y-"][z]+float(j[p+1])
                    p = p+2
    return inp

data = xychange(data,takelist)
tdata = xychange(tdata,takelist)


#拿掉提取完的string features
for i in data:
    if i in takelist:
        data = data.drop(i,axis=1)

for i in tdata:
    if i in takelist:
        tdata = tdata.drop(i,axis=1)

#把要預測的參數放到後面的位置
switchcolumn = ["Input_A6_024","Input_A3_016","Input_C_013","Input_A2_016",
                "Input_A3_017","Input_C_050","Input_A6_001","Input_C_096",
                "Input_A3_018","Input_A6_019","Input_A1_020","Input_A6_011",
                "Input_A3_015","Input_C_046","Input_C_049","Input_A2_024",
                "Input_C_058","Input_C_057","Input_A3_013","Input_A2_017"]
def swap_columns(df, c1, c2):
    cols = list(df.columns)
    a,b = cols.index(c1),cols.index(c2)
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    return df

for i,j in enumerate(switchcolumn):
    data = swap_columns(data,j,data.columns[-(i+1)])

#change test columns order
tdata = tdata.reindex(columns=data.columns[0:228])

train = data[0:300]
valid = data[300:]
train_x = train.iloc[:,1:228]
valid_x = valid.iloc[:,1:228]
train_y = train.iloc[:,228:]
valid_y = valid.iloc[:,228:]

clas = xgb.XGBRegressor()
clas.fit(train_x,train_y.iloc[:,0])
clas.score(valid_x,valid_y.iloc[:,0])
xv = clas.predict(valid_x)

rms = sqrt(mean_squared_error(valid_y.iloc[:,0], xv))