from feature_extractor import FeatureExtractor


class Model:
    def __init__(self, image_db, test_percentage=0.1):
        self.feature_database = [[], []]

        images, masks, labels, file_names = image_db
        db_labeled = {}
        for i in range(len(images)):
            if labels[i] not in db_labeled:
                db_labeled[labels[i]] = []
            db_labeled[labels[i]].append(i)

        test_images = []
        test_masks = []
        test_labels = []
        model_images = []
        model_masks = []
        model_labels = []

        for label in db_labeled:
            test_count = int(len(db_labeled[label]) * test_percentage)
            for j in range(len(db_labeled[label])):
                image = images[db_labeled[label][j]]
                mask = masks[db_labeled[label][j]]
                if j < test_count:
                    test_images.append(image)
                    test_masks.append(mask)
                    test_labels.append(label)
                    print(file_names[j])
                else:
                    model_images.append(image)
                    model_masks.append(mask)
                    model_labels.append(label)

        self.test_database = test_images, test_masks, test_labels

        for i in range(len(model_images)):
            extractor = FeatureExtractor(model_masks[i])
            vis, features = extractor.process()
            self.feature_database[0].append(model_labels[i])
            self.feature_database[1].append(features)

    def _dist(self, featuresA, featuresB):
        sqr_sum = 0
        for i in range(len(featuresA)):
            sqr_sum += (featuresA[i] - featuresB[i]) ** 2

        return sqr_sum

    def Classify(self, image):
        extractor = FeatureExtractor(image)
        vis, features = extractor.process()

        closest = 0
        min_dist = self._dist(features, self.feature_database[1][closest])
        for i in range(len(self.feature_database[1])):
            dist = self._dist(features, self.feature_database[1][i])
            if dist < min_dist:
                closest = i
                min_dist = dist

        return self.feature_database[0][closest]

    def Test(self):
        images, masks, labels = self.test_database

        correct_count = 0
        incorrect_count = 0
        for i in range(len(images)):
            label = self.Classify(masks[i])
            correct_count += label == labels[i]
            incorrect_count += label != labels[i]

        return correct_count, incorrect_count
