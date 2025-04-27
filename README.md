For VAE model: 
CelebA.py and Cifar 10.py
For Cifar 10:
Run it from the console:
Assuming your project root contains both:
Cifar 10.py
cifar-10-batches-py/      ← contains data_batch_1 … data_batch_5
cifar 10 result/         ← (empty) where you want PNG/NPZ saved
On Windows PowerShell/ consile, quote the space-containing folder:
py -3 cifar 10.py --data_dir .\cifar-10-batches-py --output_dir "cifar 10 result" --batch_size 128 --epochs 30
Edit batch size and epochs as needed
This will load all five CIFAR‑10 batches, train the VAE for 30 epochs, and every 5 epochs dump a 4×4 PNG plus a .npz of the raw reconstructions into your cifar 10 result folder.

For the metrics for Cifar 10.py
Please install the extra dependency that InceptionScore pulls in(put following comman in your console):

py -3 -m pip install torch-fidelity
py -3 -m pip install torchmetrics[image]
and run:
py -3 metrics Cifar 10.py
You should now see an IS (≫1) and FID for your “cifar 10 result” images.


For CelebA:
Your CelebA images stored in a single folder (img_align_celeba/) and dumping reconstructions into cifar a result/
py -3 CelebA.py --data_dir .\img_align_celeba --output_dir "cifar a result" --batch_size 128 --epochs 30
This will:
Read all all .jpg files in img_align_celeba/,
Center‑crop to 178×178 → resize to 64×64 → normalize to [–1,1],
Train the same ConvVAE for 30 epochs,
Every 5 epochs save a 4×4 PNG grid plus a .npz of raw reconstructions into cifar a result/.

For the metrics for CelebA.py:
Please install the extra dependency that InceptionScore pulls in(put following comman in your console):
py -3 -m pip install torch-fidelity
py -3 -m pip install torchmetrics[image]
and run:
py -3 metrics_cifaba.py --gen_folder "cifar a result" --batch_size 64
You should now see an IS (≫1) and FID for your “CelebA result” images.
