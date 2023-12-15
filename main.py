from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import sys

# load model -------------------------------------------------------------------
model = nn.Sequential(
    nn.Conv2d(3, 16, 5),
    nn.MaxPool2d((5, 5)),
    nn.LeakyReLU(),
    nn.Flatten(),
    nn.Linear(400, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 2)
)

model.load_state_dict(torch.load('model.pt'))
model.eval()


# Open the image file ----------------------------------------------------------
image = Image.open(sys.argv[1])

# Convert RGBA or BW to RGB ----------------------------------------------------
image = image.convert('RGB')

# Crop the image ---------------------------------------------------------------
transform = transforms.Compose([
    transforms.CenterCrop(min(image.size)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

image_tensor = transform(image)

plt.imshow(transforms.ToPILImage()(image_tensor))
plt.show()

# add batch dimension ----------------------------------------------------------
image_tensor = image_tensor.unsqueeze(0)

# forward pass -----------------------------------------------------------------
output = model.forward(image_tensor)
output = torch.softmax(output, dim=1)
print(output)

if output[0][0] > output[0][1]:
    print("This is a cat!")
    sys.exit(0)

else:
    print("This is not a cat!")
    sys.exit(1)
