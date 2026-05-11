from image_loader import ImageLoader
from feature_extractor import FeatureExtractor
from presenter import Presenter

loader = ImageLoader()
masks, mask_file_names = loader.load_masks_random(5)

for i in range(len(masks)):
    featureExtractor = FeatureExtractor(masks[i])
    vis_results, features = featureExtractor.process()

    presenter = Presenter()

    presenter.show(
        mask_file_names[i] + "\nDefects: " + str(len(features)), vis_results, 2
    )
