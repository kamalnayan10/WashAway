from PIL import Image , ImageOps
import os

class ImageSegmentation():
    def __init__(self , image_path , dir = "patches" , SIZE = 512 , out_img_path = "stitched_image"):

        self.image = Image.open(image_path)
        self.dir = dir
        self.img_width = self.image.width
        self.img_height = self.image.height
        self.output_image_path = out_img_path
        self.SIZE = SIZE


    def create_patches(self):

        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

        num_x = (self.img_width + self.SIZE - 1) // self.SIZE  # Ceiling division
        num_y = (self.img_height + self.SIZE - 1) // self.SIZE  # Ceiling division
        
        patch_count = 0

        for t in range(num_y):
            for l in range(num_x):
                top = t * self.SIZE
                left = l * self.SIZE
                right = min(left + self.SIZE, self.img_width)
                bottom = min(top + self.SIZE, self.img_height)

                patch = self.image.crop((left , top , right , bottom))

                if patch.size != (self.SIZE, self.SIZE):
                    patch = ImageOps.pad(patch, (self.SIZE, self.SIZE), method=Image.BICUBIC, color=(0, 0, 0))

                patch.save(f"{self.dir}/patch_{t}_{l}.jpg")

                patch_count += 1

        print(f'Total patches created: {patch_count}')

    def stitch_patches(self):

        num_x = (self.img_width + self.SIZE - 1) // self.SIZE  # Ceiling division
        num_y = (self.img_height + self.SIZE - 1) // self.SIZE  # Ceiling division

        stitched_image = Image.new('RGB', (self.img_width, self.img_height))

        for t in range(num_y):
            for l in range(num_x):
                patch = Image.open(f"{self.dir}/patch_{t}_{l}.jpg")

                top = t * self.SIZE
                left = l * self.SIZE
                right = min(left + self.SIZE, self.img_width)
                bottom = min(top + self.SIZE, self.img_height)

                side_gap = (self.SIZE - (right - left))/2

                top_gap = (self.SIZE - (bottom - top))/2

                crop_box = (side_gap , top_gap, right - side_gap, bottom - top_gap)
                patch = patch.crop(crop_box)

                stitched_image.paste(patch, (left, top))

        stitched_image.save(f"{self.output_image_path}.jpg")
        print("Image stitched back!")

if __name__ == "__main__":

    image = ImageSegmentation("watermarked.jpg" , dir = "patches_watermarked" , out_img_path="watermarked_stitched")
    image2 = ImageSegmentation("test.jpg" , dir = "patches_nonwatermarked" , out_img_path="nonwatermarked_stitched")
    image.create_patches()
    image.stitch_patches()
    image2.create_patches()
    image2.stitch_patches()