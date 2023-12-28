# Implementations for different image transformation used for data augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
import torchvision
from torchvision import transforms
import diff

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class GridMask():
    def __init__(self, grid_m=8, grid_n=8) -> None:
        self.m=grid_m
        self.n=grid_n
    
    def __call__(self, inputs):
        inputs = inputs.clone()
        for k, input in enumerate(inputs):
            h, w = input.shape[1:]  # input.shape = (3, 32, 32)
            cell_h = h // self.m
            cell_w = w // self.n
            for i in range(self.m):
                for j in range(self.n):
                    if (i+j)%2==0:
                        inputs[k][:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 0
        return inputs

class HideandSeek(GridMask):
    def __init__(self, grid_m=8, grid_n=8, p=0.4) -> None:
        super().__init__(grid_m, grid_n)
        self.p=p
    
    def __call__(self, inputs):
        inputs = inputs.clone()
        for k, input in enumerate(inputs):
            h, w = input.shape[1:]  # input.shape = (3, 32, 32)
            cell_h = h // self.m
            cell_w = w // self.n
            for i in range(self.m):
                for j in range(self.n):
                    prob=np.random.rand()
                    if (i+j)%2==0 and prob<self.p:
                        inputs[k][:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 0
        return inputs

# class Saliency_Map(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         n, c, h, w = x.size()
#         assert h == w
        
#         # We don't need gradients w.r.t. weights for a trained model
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#         # Set the model in eval mode
#         self.model.eval()
        
#         # Create a transformation to convert the tensor to a PIL Image and normalize it
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         ret = x.clone()
#         for k, img in enumerate(x):
#             # Transform the input tensor to a PIL Image and normalize it
#             img = img[:3, :, :]
#             input = transform(img)
            
#             input.unsqueeze_(0)
            
#             # We want to calculate the gradient of the highest score w.r.t. input
#             # so set requires_grad to True for input
#             input.requires_grad = True
            
            
#             # Forward pass to calculate predictions
#             preds = self.model(input)
#             score, indices = torch.max(preds, 1)
            
#             # Backward pass to get gradients of the score of the predicted class w.r.t. input image
#             score.backward()
            
#             # Get the maximum values along the channel axis
#             slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
            
#             # Normalize to [0..1]
#             slc = (slc - slc.min()) / (slc.max() - slc.min())
#             slc = F.interpolate(slc.unsqueeze(0).unsqueeze(0), (h, w), mode='bilinear', align_corners=False)
#             slc = slc[0].repeat(c,1,1)
#             ret[k] = slc
        
#         return ret

class Saliency_Map(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        
        # We don't need gradients w.r.t. weights for a trained model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set the model in eval mode
        self.model.eval()
        
        
        x_first_3_channel = x[:, :3, :, :]
        x_first_3_channel.requires_grad = True
        preds = self.model(self.transform(x_first_3_channel))
        # print(preds.shape)
        score, indices = torch.max(preds, 1)
        # print(score.shape)
        score.backward(torch.ones_like(score))
        # print(x_first_3_channel.grad.shape)
        slc = torch.zeros((n, 1, h, w))
        for i in range(n):
            slc[i] = torch.max(torch.abs(x_first_3_channel.grad[i]), dim=0)[0].unsqueeze(0)
        slc = (slc - slc.min()) / (slc.max() - slc.min())
        slc = F.interpolate(slc, (h, w), mode='bilinear', align_corners=False)
        slc = slc.repeat(1, c, 1, 1)
        # print(slc.shape)

        return slc.to('cuda')

class MaxCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        kernel = torch.ones(1, c, int(self.size*h), int(self.size*w)).to(x.device)
        y = x.clone()
        for i, img in enumerate(x):
            img = img.unsqueeze(0)
            result = F.conv2d(img, kernel, stride=1, padding=0)
            max_value, max_index = torch.max(result.view(1, -1), dim=1)
            max_position = np.unravel_index(max_index.item(), result.shape[2:])
            corresponding_part = img[:, :, max_position[0]:max_position[0] + int(0.8*h), max_position[1]:max_position[1] + int(0.8*w)]
            resized_corresponding_part = F.interpolate(corresponding_part, size=(h, w), mode='nearest')
            y[i] = resized_corresponding_part[0]
        return y

class SeqAug(nn.Module):
    def __init__(self, aug1, aug2):
        super().__init__()
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, x):
        x1 = self.aug1(x)
        x2 = self.aug2(x1)
        return x2

class DPT(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoImageProcessor, DPTForDepthEstimation
        self.model_path = '/bigdata/users/ve490-fall23/lyf/drqv2/.cache/huggingface/models--Intel--dpt-hybrid-midas/snapshots/530c4509f775419d716ca24ad1e7c6095e8ac2b4'
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = DPTForDepthEstimation.from_pretrained(self.model_path).to('cuda')

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        y = x.clone()
        for i, img in enumerate(x):
            # print(img.shape)
            img = img[:3, :, :]
            transform = transforms.ToPILImage()
            img = transform(img)
            inputs = self.image_processor(images=img, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # visualize the prediction
            resized_tensor = F.interpolate(prediction, size=(h, w), mode='bilinear', align_corners=False)
            # print(resized_tensor.shape)
            y[i] = resized_tensor
        return y

class Diffeo(torch.nn.Module):
    """Randomly apply a diffeomorphism to the image(s).
    The image should be a Tensor and it is expected to have [..., n, n] shape,
    where ... means an arbitrary number of leading dimensions.
    
    A random cut is drawn from a discrete Beta distribution of parameters
    alpha and beta such that
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards cutmax the distribution is)
        
    Given cut and the allowed* interval of temperatures [Tmin, Tmax], a random T is
    drawn from a Beta distribution with parameters alpha and beta such that:
        s = alpha + beta (measures how peaked the distribution is)
        r = alpha / beta (measured how biased towards T_max the distribution is)

    Beta ~ delta_function for s -> inf. To apply a specific value x \in [0, 1]
    in the allowed interval of T or cut, set
        - s = 1e10
        - r = x / (1 - x)

    *the allowed T interval is defined such as:
        - Tmin corresponds to a typical displacement of 1/2 pixel in the center
          of the image
        - Tmax corresponds to the highest T for which no overhangs are present.

    Args:
        sT (float):  
        rT (float): 
        scut (float):  
        rcut (float): 
        cut_min (int): 
        cut_max (int): 
        
    Returns:
        Tensor: Diffeo version of the input image(s).

    """
    

    def __init__(self, sT=25.0, rT=0.5, scut=50.0, rcut=1.0, cutmin=1, cutmax=10):
        super().__init__()
        
        self.sT = sT
        self.rT = rT
        self.scut = scut
        self.rcut = rcut
        self.cutmin = cutmin
        self.cutmax = cutmax
        
        self.betaT = torch.distributions.beta.Beta(sT - sT / (rT + 1), sT / (rT + 1), validate_args=None)
        self.betacut = torch.distributions.beta.Beta(scut - scut / (rcut + 1), scut / (rcut + 1), validate_args=None)
    
    def forward(self, img):
        """
        Args:
            img (Tensor): Image(s) to be 'diffeomorphed'.

        Returns:
            Tensor: Diffeo image(s).
        """
        
        # image size
        n = img.shape[-1]
        
        cut = (self.betacut.sample() * (self.cutmax + 1 - self.cutmin) + self.cutmin).int().item()
        T1, T2 = diff.temperature_range(n, cut)
        T = (self.betaT.sample() * (T2 - T1) + T1)
        
        return diff.deform(img, T, cut)
    

    def __repr__(self):
        return self.__class__.__name__ + f'(sT={self.sT}, rT={self.rT}, scut={self.scut}, rcut={self.rcut}, cutmin={self.cutmin}, cutmax={self.cutmax})'

class HalfHalfAug(nn.Module):
    def __init__(self, aug1, aug2):
        super().__init__()
        self.aug1 = aug1
        self.aug2 = aug2
    
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        mid = n // 2
        x1 = self.aug1(x[:mid])
        x2 = self.aug2(x[mid:])
        return torch.cat([x1, x2], dim=0)

class DataAug(nn.Module):
    def __init__(self, da_type):
        super().__init__()
        self.data_aug_type = da_type
        if self.data_aug_type == 1:
            # random shift
            self.aug = RandomShiftsAug(4)
        elif self.data_aug_type == 2:
            self.aug = kornia.augmentation.RandomRotation(degrees=180., p=1., same_on_batch=False)
        elif self.data_aug_type == 3:
            self.aug = kornia.augmentation.RandomVerticalFlip(p=.5, same_on_batch=False)
        elif self.data_aug_type == 4:
            self.aug = kornia.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=False)
        elif self.data_aug_type == 5:
            self.aug = GridMask(grid_m=8, grid_n=8)
        elif self.data_aug_type == 6:
            self.aug = HideandSeek(grid_m=8, grid_n=8, p=0.4)
        elif self.data_aug_type == 7:
            # model = torchvision.models.resnet50(pretrained=True).to('cuda')
            model = torchvision.models.vgg19(pretrained=True).to('cuda')
            self.aug = Saliency_Map(model)
        elif self.data_aug_type == 8:
            self.aug = kornia.augmentation.RandomAffine(degrees=180, p=1, same_on_batch=False, translate=(0.5, 0.5))
        elif self.data_aug_type == 9:
            self.aug = kornia.augmentation.RandomRotation(degrees=5., p=1., same_on_batch=False)
        elif self.data_aug_type == 10:
            self.aug = kornia.augmentation.RandomAffine(degrees=5., p=1, same_on_batch=False, translate=(0.5, 0.5))
        elif self.data_aug_type == 11:
            self.aug = kornia.augmentation.RandomRotation(degrees=5., p=1., same_on_batch=False)
        elif self.data_aug_type == 12:
            self.aug = kornia.augmentation.CenterCrop(size=(59, 59), p=1., keepdim=True, cropping_mode='resample')
        elif self.data_aug_type == 13:
            self.aug = kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
        elif self.data_aug_type == 14:
            self.aug = MaxCrop(size=0.7)
        elif self.data_aug_type == 15:
            self.aug = kornia.augmentation.container.AugmentationSequential(
                kornia.augmentation.RandomRotation(degrees=180., p=1., same_on_batch=False),
                kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
            )
        elif self.data_aug_type == 16:
            aug1 = kornia.augmentation.RandomRotation(degrees=180., p=1., same_on_batch=False)
            aug2 = MaxCrop(size=0.7)
            self.aug = SeqAug(aug1, aug2)
        elif self.data_aug_type == 17:
            self.aug = DPT()
        elif self.data_aug_type == 18:
            aug1 = RandomShiftsAug(4)
            aug2 = kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
            self.aug = SeqAug(aug2, aug1)
        elif self.data_aug_type == 19:
            aug1 = RandomShiftsAug(4)
            aug2 = Diffeo()
            self.aug = SeqAug(aug2, aug1)
        elif self.data_aug_type == 20:
            aug1 = RandomShiftsAug(4)
            aug2 = Diffeo()
            self.aug = SeqAug(aug1, aug2)
        elif self.data_aug_type == 21:
            aug1 = RandomShiftsAug(4)
            aug2 = Diffeo()
            self.aug = kornia.augmentation.container.AugmentationSequential(
                aug1,
                aug2
            )
        elif self.data_aug_type == 22:
            aug1 = RandomShiftsAug(4)
            aug2 = kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
            self.aug = SeqAug(aug1, aug2)
        elif self.data_aug_type == 23:
            aug1 = RandomShiftsAug(4)
            aug2 = kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
            self.aug = kornia.augmentation.container.AugmentationSequential(
                aug1,
                aug2
            )
        elif self.data_aug_type == 24:
            aug1 = RandomShiftsAug(4)
            aug2 = Diffeo()
            self.aug = HalfHalfAug(aug1, aug2)
        elif self.data_aug_type == 25:
            aug1 = RandomShiftsAug(4)
            aug2 = kornia.augmentation.RandomElasticTransform(kernel_size=(21, 21), p=1., same_on_batch=False, keepdim=True)
            self.aug = HalfHalfAug(aug1, aug2)
        elif self.data_aug_type.startswith('rset_'):
            kernel = int(self.data_aug_type.strip('rset_'))
            aug1 = RandomShiftsAug(4)
            aug2 = kornia.augmentation.RandomElasticTransform(kernel_size=(kernel, kernel), p=1., same_on_batch=False, keepdim=True)
            self.aug = kornia.augmentation.container.AugmentationSequential(
                aug1,
                aug2
            )
        elif self.data_aug_type.startswith('rsdiff_'):
            params = self.data_aug_type.split('_')
            sT, rT = eval(params[1]), eval(params[2])
            aug1 = RandomShiftsAug(4)
            aug2 = Diffeo(sT=sT, rT=rT)
            self.aug = kornia.augmentation.container.AugmentationSequential(
                aug1,
                aug2
            )

    def forward(self, x):
        return self.aug(x)