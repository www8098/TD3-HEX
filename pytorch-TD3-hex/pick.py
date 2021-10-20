import pickle
import numpy as np

path = 'data/walker2d-expert-v2.pkl'
f = open(path, 'rb')
data = pickle.load(f)
id = np.random.randint(0, 1000, 100)
trans = []
for i in id:
    trans.append(data[i])
with open('data/walker2d-poor.pkl', 'wb') as f:
    pickle.dump(trans, f)

# with open('data/halfcheetah-poor.pkl', 'rb') as f:
#     data = pickle.load(f)
#     acc_r = []
#     for traj in data:
#         acc_r.append(sum(traj['rewards']))
#     print(acc_r)
# acc_r = enumerate(acc_r)
# acc_r.sorted(key = lambda x:x[1])
# print(acc_r[:10])
# max_id = acc_r.index(max(acc_r))
# print(data[1].keys())
# print(len(data[1]))