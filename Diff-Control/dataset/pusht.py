import torch
from torchvision import transforms
from einops import rearrange
import clip

class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, pusht_dataset, image_size=(224, 224)):
        self.pusht_dataset = pusht_dataset
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
                                   )
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.dummy_sentence = "push the T to its goal"
        self.transform = transforms.Resize(self.image_size)

    def __len__(self):
        return len(self.square_ph_dataset)

    def __getitem__(self, idx):
        transition_history = 12
        data = self.pusht_dataset.__getitem__(idx)
        if idx < transition_history:
            idx_pre = 0
        else:
            idx_pre = idx - transition_history

        prev_data = self.pusht_dataset.__getitem__(idx_pre)

        # Extract image and action from the PushT dataset
        image = data['obs']['agentview_image']
        action = rearrange(data['action'], "B Ta Da -> B Da Ta")

        prior_action = rearrange(prev_data['action'], "Ta Da -> Da Ta")
        # Resize and normalize the image to match Duck dataset
        image = self.transform(image)

        # Tokenize dummy sentence using CLIP
        sentence = clip.tokenize([self.dummy_sentence])[0]

        return image, prior_action, action, sentence

    def get_validation_dataset(self):
        val_set = self.pusht_dataset.get_validation_dataset()
        return PushTDataset(val_set, image_size=self.image_size)
