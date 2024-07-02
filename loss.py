import torch.nn as nn
import torch 
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import config

# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval().to(config.device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class Edge_Gradient_loss(nn.Module):
    def __init__(self):
        super(Edge_Gradient_loss,self).__init__()
        
    def sobel_filter(self,image):
        """
        Apply Sobel filter to the input image to get the gradients.
        :param image: Input image tensor of shape (B, C, H, W)
        :return: Gradient in x direction and y direction
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        
        sobel_x = sobel_x.to(config.device)
        sobel_y = sobel_y.to(config.device)

        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)

        return grad_x, grad_y

    def gradient_variance_loss(self,output, target):
        """
        Compute the Gradient Variance Loss between the output and target images.
        :param output: Generator produced image tensor of shape (B, C, H, W)
        :param target: Original high-resolution image tensor of shape (B, C, H, W)
        :return: Gradient Variance Loss
        """
        grad_x_output, grad_y_output = self.sobel_filter(output)
        grad_x_target, grad_y_target = self.sobel_filter(target)

        var_x_output = torch.var(grad_x_output)
        var_y_output = torch.var(grad_y_output)
        var_x_target = torch.var(grad_x_target)
        var_y_target = torch.var(grad_y_target)

        gv_loss = torch.abs(var_x_output - var_x_target) + torch.abs(var_y_output - var_y_target)

        return gv_loss

    def forward(self, output, target):
        return self.gradient_variance_loss(output, target)
    
if __name__ == "__main__":
    output_image = torch.randn(1, 1, 510, 510)  # Generator output
    target_image = torch.randn(1, 1, 510, 510)  # Original high-resolution image

    # Move tensors to GPU if available
    output_image = output_image.to(config.device)
    target_image = target_image.to(config.device)

    # Compute GV loss
    loss = Edge_Gradient_loss()
    l = loss(output_image, target_image)
    print("Gradient Variance Loss:", l.item())