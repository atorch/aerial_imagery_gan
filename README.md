# aerial_imagery_gan

A GAN for aerial imagery

The inputs to the model are
[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/)
(NAIP) scenes. These are four band (R G B NIR) aerial imagery rasters downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
and placed in the `./naip` directory.

```bash
sudo docker build ~/aerial_imagery_gan --tag=aerial_imagery_gan
sudo docker run --gpus all -it -v ~/aerial_imagery_gan:/home/aerial_imagery_gan aerial_imagery_gan bash
python src/fit_gan.py
```

Real NAIP image patches look like this:

![Sample NAIP patch 0](examples/real_image_0.png)

![Sample NAIP patch 1](examples/real_image_1.png)

![Sample NAIP patch 2](examples/real_image_2.png)

![Sample NAIP patch 3](examples/real_image_3.png)

![Sample NAIP patch 4](examples/real_image_4.png)

![Sample NAIP patch 5](examples/real_image_5.png)

The generator output (shown here for a fixed epoch, with varying input noise)
looks like this:

![Generator output 0](generator_output/generated_image_noise_0_epoch_384.png)

![Generator output 1](generator_output/generated_image_noise_1_epoch_384.png)

![Generator output 2](generator_output/generated_image_noise_2_epoch_384.png)

Since the generator is fully convolutional, we can also use it to generate
images larger than the patches it was trained on, like this:

![Generator large output](generator_output/large_image_epoch_384.png)