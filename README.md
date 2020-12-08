# aerial_imagery_gan

A GAN for aerial imagery

```bash
sudo docker build ~/aerial_imagery_gan --tag=aerial_imagery_gan
sudo docker run --gpus all -it -v ~/aerial_imagery_gan:/home/aerial_imagery_gan aerial_imagery_gan bash
python src/fit_gan.py
```
