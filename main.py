from image_loader import ImageLoader
from feature_extractor import FeatureExtractor
from presenter import Presenter
from model import Model

loader = ImageLoader()
images, masks, file_names, labels = loader.load_all()

correct_sum = 0
incorrect_sum = 0
for i in range(40):
    model = Model((images, masks, labels), test_percentage=0.1)
    correct_count, incorrect_count = model.Test()
    correct_sum += correct_count
    incorrect_sum += incorrect_count
    # print(correct_count, incorrect_count)
    # print(correct_count / (incorrect_count + correct_count) * 100)

print(correct_sum, incorrect_sum)
print(correct_sum / (incorrect_sum + correct_sum) * 100)
