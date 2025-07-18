from diffusion_model import DiffusionModel  # replace with actual filename

# 1. Load your trained diffusion model
diffusion_model = DiffusionModel(device="cpu", 
                                     checkpoint_name="_checkpoint_138.pth", mode="test")

# 2. Generate synthetic samples and contexts
generated_images = diffusion_model.test(batch_size=1)

