import VGG_16
import data
from sklearn.model_selection import train_test_split

d = data.Data()
inputExamples = d.inputExamples
train_examples, dev_examples = train_test_split(inputExamples, random_state=42, test_size=0.2)
model = VGG_16.Model(train_examples, dev_examples)
model.train()