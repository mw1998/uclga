import torch 
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from torch.utils.data import Dataset
from skimage import io 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class MyDataset(Dataset):
    def __init__(self, dirs, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.dirs = dirs
        self.annotations = os.listdir(os.path.join(root_path, dirs))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.dirs, self.annotations[index])
        img = io.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.annotations[index]

my_transforms = transforms.Compose([
    transforms.ToPILImage(), # all the transformations work on this format
    # transforms.Resize((256, 256)),
    # transforms.RandomCrop((200, 200)),
    # transforms.CenterCrop((128, 128)),
    # transforms.RandomAffine(degrees=45),
    # transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # (value - mean) / std, this does noting!
    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])

test_transorms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.CenterCrop((256, 256)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

classtype = 'ellipse'
dataset = MyDataset(classtype, r'D:\retrain_classification_kfold3_new2\newdataset2\newotrain2', transform=my_transforms)
print(len(dataset))
save_path = './newdataset2/train_2/' + classtype + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_num = 0
for _ in range(7):
    for img, name in dataset:
        save_image(img, save_path + classtype + str(img_num) + '.png')
        # save_image(img, save_path +  str(name))
        img_num += 1