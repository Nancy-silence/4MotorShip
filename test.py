# from collections import defaultdict, OrderedDict
# import json
# # video = defaultdict(list)
# # video["label"].append("haha")
# # video["data"].append(234)
# # video["score"].append(0.3)
# # video["label"].append("xixi")
# # video["data"].append(123)
# # video["score"].append(0.7)
# test_dict = {
#     '1': "1.0",
#     #'results': video,
#     'explain': {
#         'used': True,
#         'details': "this is for josn test",
#   }
# }

# json_str = json.dumps(test_dict)
# with open('test_data.json', 'w') as json_file:
#     json_file.write(json_str)

# import json
# xy = {'5': [], '6': [], '10': [], '11': [], '50': [], '51': [], '500':[], '501': [], '1000':[], '1001': []}
# xy['%i' % (5)].append(s[0])

# filename2='xy.json'
# json_str_xy = json.dumps(xy)
# with open(filename2,'w') as file_obj2:
#     file_obj2.write(json_str_xy)



# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)

# def write_json(filename, content):
#     """
#     @param filname:存储的json文件名称
#     @param content:要存储的python数据，一般为List
#     """
#     temp = json.dumps(content, cls=NpEncoder)
#     with open(filename,'w') as file_obj:
#         file_obj.write(temp)


    # write_json('reward.json', reward)
    # write_json('x.json', x)
    # write_json('y.json', y)
    # write_json('error.json', error)
    # write_json('td_error.json', td_error)
    # write_json('loss_a.json', loss_a)

import numpy as np
import math
# a=(np.random.rand(2)-0.5)/5
# print(a)
# a = np.pad(a,(0,5),'constant',constant_values=(0,0))
# print(a)
# b = np.array([0,0,1,1,1,1,1])
# print(b)
# print(a+b)
print(math.tanh(0.04))