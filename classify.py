from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python classify.py path_to_image [more paths ...]")
    exit()

# load model -------------------------------------------------------------------
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.MaxPool2d((3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(64, 256, 3),
    nn.MaxPool2d((3, 3)),
    nn.LeakyReLU(),
    nn.Flatten(),
    nn.Linear(1024, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 2)
)

model.load_state_dict(torch.load('model.pt'))
model.eval()


for i in range(1, len(sys.argv)):
    # Open the image file ----------------------------------------------------------
    image = Image.open(sys.argv[i])

    # Convert RGBA or BW to RGB ----------------------------------------------------
    image = image.convert('RGB')

    # Crop the image ---------------------------------------------------------------
    transform = transforms.Compose([
        transforms.CenterCrop(min(image.size)),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image)

    # add batch dimension ----------------------------------------------------------
    image_tensor = image_tensor.unsqueeze(0)

    # forward pass -----------------------------------------------------------------
    output = model.forward(image_tensor)
    output = torch.softmax(output, dim=1)
    print(output)

    if output[0][0] > output[0][1]:
        message = "This is a cat!"

    else:
        message = "This is not a cat!"

    print(message)
    confidence = round(float(max(output[0]) * 100), 2)
    plt.title(f'{message} Confidence: {confidence}%')
    plt.imshow(transforms.ToPILImage()(image_tensor[0]))
    plt.show()
