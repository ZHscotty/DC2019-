import VGG_16
import data
import pickle
import dense
import numpy as np
import pandas as pd
import config
import VGG
import CNN
from sklearn.model_selection import train_test_split

######################################################################################
# 重新处理数据
d = data.Data('E:\ZH project\data\新网银行唇语识别竞赛数据')
lid2label = d.lid2label
inputExamples = d.load_example()
testExamples = d.load_test()
with open('../result/examples/test_example_{}.pkl'.format(config.WIDTH), 'wb') as f:
    pickle.dump(testExamples, f)
    print('save test_example_{} ok'.format(config.WIDTH))

with open('../result/examples/input_example_{}.pkl'.format(config.WIDTH), 'wb') as f:
    pickle.dump(inputExamples, f)
    print('save input_example_{} ok!'.format(config.WIDTH))
#######################################################################################


#######################################################################################
# 直接加载保存好的数据
# with open('../result/examples/test_example_64.pkl', 'rb') as f:
#     testExamples = pickle.load(f)
#
#
# with open('../result/examples/input_example_64.pkl', 'rb') as f:
#     inputExamples = pickle.load(f)
#######################################################################################
train_examples, dev_examples = train_test_split(inputExamples, random_state=42, test_size=0.2)
model = VGG.Model()
model.train(train_examples, dev_examples)
predict = model.predict(dev_examples)
print(predict.shape)
predict = np.argmax(predict, axis=1)

output = {}
id = []
result = []
for index in range(len(dev_examples)):
    id.append(dev_examples[index].id)
    result.append(lid2label[predict[index]])
output['id'] = id
output['word'] = result
out = pd.DataFrame(output)
out.to_csv('../result/submit/10_16.csv', index=False)
print('output ok')