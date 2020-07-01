import ctypes
from ctypes import *
import numpy as np
import math

class StructPointer(Structure):
    """
    在python中创建struct类来与c++中的struct相对应
    通过_fields_指定struct的内容
    """
    # list[tuple("key", value_type)]
    _fields_ = [("array", c_double * 10)]

def step(cur_state, action, time_interval):

    # 通过CDLL加载.so文件为python的module
    sim = CDLL('./Sim.dll')

    # 通过标准方法将python list 转为 c 的数组
    pyarray = np.append(cur_state,action).tolist()
    x = (c_double * len(pyarray))(*pyarray)
    # time
    time = c_double(time_interval)

    # 指定函数返回类型，否则会默认为int
    sim.my_integrate.restype = POINTER(StructPointer)

    # 获取返回值
    result = sim.my_integrate(x, time)

    # 获取返回值中的array，取前6项为状态，注意.contents
    ret_array = result.contents.array[0:6]
    # print(f'theta_init: {ret_array[2]}')
    # 对φ角进行修正，到范围[-pi,pi]内
    remainder_fai = ret_array[2] % (2 *math.pi)
    if abs(remainder_fai) <= math.pi:
        fai = remainder_fai
    else:
        fai = -(2 * math.pi - abs(remainder_fai)) if remainder_fai > 0 else 2 * math.pi - abs(remainder_fai)
    ret_array[2] = fai

    # for param in ret_array:
    #     print(param)

    return ret_array

# 程序测试
'''
import xlrd
import json
class excel_read:
    def __init__(self, excel_path=r'/root/ASV/c_env/action.xlsx',encoding='utf-8',index=0):

      self.data=xlrd.open_workbook(excel_path)  ##获取文本对象
      self.table=self.data.sheets()[index]     ###根据index获取某个sheet
      self.rows=self.table.nrows   ##3获取当前sheet页面的总行数,把每一行数据作为list放到 list

    def get_data(self):
        result=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  ##获取每一列数据
            result.append(col)
        print(result)
        return result

if __name__ == '__main__':
    #读入action数据，调用step()计算state数据
    action_list = excel_read().get_data()
    state_list = []
    for i in range(len(action_list)):
        if i==0:
            state_list.append([1,4.1,0.5,0.01,0.01,0.01])
        state = step(state_list[i],action_list[i])
        state_list.append(state)
    print(state_list)

    #存储state_list为json文件
    filename='state_cal.json'
    with open(filename,'w') as file_obj:
        json.dump(state_list,file_obj)
'''
if __name__ == '__main__':
    s = step([1.7374010205043304, 1.6136686929610684, 2.6266826787292694, -0.06877022018245621, -0.0031953394070718456, 5.732540683526411],[0.99852693, -9.626288, -8.389106, -10.0])
    print(s)
    v = math.sqrt(np.power(s[0],2) + np.power(s[1],2))/0.1
    print(f'v:{v}')







