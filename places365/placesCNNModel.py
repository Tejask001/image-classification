import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

class PlacesCNN:
    def __init__(self, arch='resnet18'):
        """
        Initializes the PlacesCNN model.

        Args:
            arch (str, optional): The architecture to use (e.g., 'resnet18', 'resnet50'). Defaults to 'resnet18'.
        """
        self.arch = arch
        self.model = self._load_model()
        self.centre_crop = self._load_image_transformer()
        self.classes = self._load_classes()

    def _load_model(self):
        """Loads the pre-trained Places365 model."""
        model_file = '%s_places365.pth.tar' % self.arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            print(f"Downloading model weights from {weight_url}...")
            os.system('wget ' + weight_url)

        model = models.__dict__[self.arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_image_transformer(self):
        """Creates the image transformer for preprocessing."""
        return trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_classes(self):
        """Loads the class labels from the categories_places365.txt file."""
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            print(f"Downloading class labels from {synset_url}...")
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        return tuple(classes)

    def detect(self, image_path, topk=5):
        """
        Detects the scene category in the given image.

        Args:
            image_path (str): The path to the image file.
            topk (int, optional): The number of top predictions to return. Defaults to 5.

        Returns:
            list: A list of tuples, where each tuple contains the probability and class label
            for the top k predictions.
        """
        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            return "Error: Image not found at {}".format(image_path)
        except Exception as e:
            return "Error: Could not open image: {}".format(e)

        input_img = V(self.centre_crop(img).unsqueeze(0))

        # forward pass
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        results = []
        for i in range(0, topk):
            results.append((probs[i].item(), self.classes[idx[i]]))

        return results

if __name__ == '__main__':
    # Example usage:
    cnn = PlacesCNN()  # Initialize the PlacesCNN model

    # Download example image
    img_name = '2.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        print(f"Downloading example image from {img_url}...")
        os.system('wget ' + img_url)

    predictions = cnn.detect(img_name)  # Detect the scene in the image

    if isinstance(predictions, str):
        print(predictions)
    else:
        print('{} prediction on {}'.format(cnn.arch, img_name))
    for prob, class_label in predictions:
        print('{:.3f} -> {}'.format(prob, class_label))