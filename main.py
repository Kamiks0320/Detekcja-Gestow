from image_loader import ImageLoader
from feature_extractor import FeatureExtractor
from presenter import Presenter
from model import Model

loader = ImageLoader()
images, masks, file_names, labels = loader.load_all()

model = Model((images, masks, labels))
# print(model.feature_database[0])
# print(model.feature_database[1])

correct_count, incorrect_count = model.Test()
print(correct_count, incorrect_count)
