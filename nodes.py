from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

def tensor2pil(image):
    """Convert a tensor to PIL Image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(image)))
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert a PIL Image to tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected PIL.Image.Image, got {}".format(type(image)))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Use alpha if available
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]
    # Convert RGB to grayscale
    return TF.rgb_to_grayscale(t.permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

class image_concat_mask:
    """Node for concatenating two images horizontally and generating a corresponding mask."""
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat_mask"
    CATEGORY = "hhy"

    def image_concat_mask(self, image1, image2=None, mask=None):
        """
        Concatenate two images horizontally and generate a mask.
        
        Args:
            image1: First image tensor (required)
            image2: Second image tensor (optional)
            mask: Input mask tensor (optional)
            
        Returns:
            tuple: (concatenated image tensor, mask tensor)
        """
        if not isinstance(image1, torch.Tensor):
            raise TypeError("image1 must be a torch.Tensor")
        
        if image2 is not None and not isinstance(image2, torch.Tensor):
            raise TypeError("image2 must be a torch.Tensor")
            
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
            
        processed_images = []
        masks = []
        
        try:
            for idx, img1 in enumerate(image1):
                # Convert tensor to PIL
                pil_image1 = tensor2pil(img1)
                
                # Get first image dimensions
                width1, height1 = pil_image1.size
                
                if image2 is not None and idx < len(image2):
                    # Use provided second image
                    pil_image2 = tensor2pil(image2[idx])
                    width2, height2 = pil_image2.size
                    
                    # Resize image2 to match height of image1
                    new_width2 = int(width2 * (height1 / height2))
                    pil_image2 = pil_image2.resize((new_width2, height1), Image.Resampling.LANCZOS)
                else:
                    # Create white image with same dimensions as image1
                    pil_image2 = Image.new('RGB', (width1, height1), 'white')
                    new_width2 = width1
                
                # Create new image to hold both images side by side
                combined_image = Image.new('RGB', (width1 + new_width2, height1))
                
                # Paste both images
                combined_image.paste(pil_image1, (0, 0))
                combined_image.paste(pil_image2, (width1, 0))
                
                # Convert combined image to tensor
                combined_tensor = pil2tensor(combined_image)
                processed_images.append(combined_tensor)
                
                # Create mask (0 for left image area, 1 for right image area)
                final_mask = torch.zeros((1, height1, width1 + new_width2))
                final_mask[:, :, width1:] = 1.0  # Set right half to 1
                
                # If mask is provided, subtract it from the right side
                if mask is not None and idx < len(mask):
                    input_mask = mask[idx]
                    # Resize input mask to match height1
                    pil_input_mask = tensor2pil(input_mask)
                    pil_input_mask = pil_input_mask.resize((new_width2, height1), Image.Resampling.LANCZOS)
                    resized_input_mask = pil2tensor(pil_input_mask)
                    
                    # Subtract input mask from the right side
                    final_mask[:, :, width1:] *= (1.0 - resized_input_mask)
                
                masks.append(final_mask)
                
            processed_images = torch.cat(processed_images, dim=0)
            masks = torch.cat(masks, dim=0)
            
            return (processed_images, masks)
        
        except Exception as e:
            raise RuntimeError(f"Error processing images: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "image concat mask": image_concat_mask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image concat mask": "Image Concat with Mask"
}