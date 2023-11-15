from torch import nn
from pathlib import Path
from fastai.vision.all import ImageDataLoaders, Resize, accuracy, vision_learner, resnet34, load_learner

if __name__ == '__main__':
    path = Path('.')
    data = ImageDataLoaders.from_folder(path, train='train', valid='valid', test='test', size=224, bs=64,
                                        num_workers=8,
                                        item_tfms=Resize(224, 224))
    print(data.vocab)
    print(len(data.train_ds), len(data.valid_ds))

    options = input("Do you want to train the model? (y/n/series): ")
    if options == "y":
        learn = vision_learner(data, resnet34, metrics=accuracy)
        learn.fine_tune(1)
        learn.export('export.pkl')
        learn = load_learner('export.pkl')
    if options == "series":
        try:
            learn = load_learner('export.pkl')
            for i in range(100):
                try:
                    print(learn.predict(f'test/cats/cat_{i}.jpg'))
                except FileNotFoundError:
                    continue
            for j in range(100):
                try:
                    print(learn.predict(f'test/dogs/dog_{j}.jpg'))
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            print("First you need program to train the model.")
    else:
        learn = load_learner('export.pkl')
        try:
            print(learn.predict('test/dogs/dog_28.jpg'))
        except FileNotFoundError:
            print('File of cat/dog not found.')
