import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

data = pd.read_csv("0714train.csv")
data["x+"]=0
data["x-"]=0
data["y+"]=0
data["y-"]=0

#data.columns.get_loc("Input_C_082")

#補缺失值#string補xy都沒移動#數值補平均值
cols = list(data.columns)
for i,j in enumerate(data.isnull().any()):
    if j:
        if data[cols[i]].dtype == "object":
            for x,z in enumerate(data[cols[i]]):
                if type(z) == float:
                    data.iloc[x,i]="N;0;N;0"
        else:
            for x,z in enumerate(data[cols[i]]):
                if np.isnan(z):
                    data.iloc[x,i]=np.mean(data[cols[i]])

#提取string(c15-c38,c63-c82)資料
for i in data.iloc[:,159:183]:    
    for z,j in enumerate(data[i]):
        if type(j) == float:
            continue
        j = j.split(";")
        p = 0
        while p < 4:
            if j[p] == "R":
                data["x+"][z] = data["x+"][z]+float(j[p+1])
            if j[p] == "L":
                data["x-"][z] = data["x-"][z]+float(j[p+1])
            if j[p] == "U":
                data["y+"][z] = data["y+"][z]+float(j[p+1])
            if j[p] == "D":
                data["y-"][z] = data["y-"][z]+float(j[p+1])
            p = p+2

for i in data.iloc[:,207:226]:    
    for z,j in enumerate(data[i]):
        if type(j) == float:
            continue
        j = j.split(";")
        p = 0
        while p < 4:
            if j[p] == "R":
                data["x+"][z] = data["x+"][z]+float(j[p+1])
            if j[p] == "L":
                data["x-"][z] = data["x-"][z]+float(j[p+1])
            if j[p] == "U":
                data["y+"][z] = data["y+"][z]+float(j[p+1])
            if j[p] == "D":
                data["y-"][z] = data["y-"][z]+float(j[p+1])
            p = p+2
                
#拿掉提取完的string features
data.columns.get_loc("Input_C_082")
data.drop(data.columns[159:183].append(data.columns[207:226]),axis=1)

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
    

