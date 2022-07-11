import argparse
import random

import PIL
import torch
import lpips
from torchvision import transforms
import os
import dnnlib
from PIL import Image
import numpy as np
from numpy.random import default_rng


import legacy

def assign_to_cluster_centers(n_samples, outdir, center_folder, network_pkl, seed=42, batch_size=32):
    """

    Args:
        n_samples: Number of sample to generate
        outdir: where to save generated images for each cluster
        center_folder: where to find each center of cluster as images
        network_pkl: Saved pkl model to generate images
        seed: Random seed used to generate latent codes
        batch_size: Size of batch when evaluating

    Returns: return centers images tensor, generated images file list and generated images center

    Generate n_samples images and assign them to a cluster and saves the generated image in outdir

    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    preprocess_lpips = transforms.Compose([
        transforms.Resize([G.img_resolution, G.img_resolution]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    all_z = torch.from_numpy(np.random.RandomState(seed).randn(n_samples, G.z_dim)).to(device).split(batch_size)
    all_c = torch.zeros([n_samples]).split(batch_size)

    os.makedirs(outdir, exist_ok=True)

    generated_dir = os.path.join(outdir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (z, c) in enumerate(zip(all_z, all_c)):
            images = G(z, c)
            images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            for idx_img, img in enumerate(images):
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
                    os.path.join(generated_dir, f'{idx * batch_size + idx_img:06d}.png'))

        del G, all_z, all_c
        # Calculating LPIPS distances to center

        centers = [preprocess_lpips(Image.open(os.path.join(center_folder, f))).to(device) for f in
                   os.listdir(center_folder)]
        num_centers = len(centers)
        centers = torch.cat(centers, dim=0)

        images_centers = torch.zeros([n_samples], dtype=torch.long).to(device)

        images_files = os.listdir(generated_dir)
        for idx, file in enumerate(images_files):
            img = preprocess_lpips(Image.open(os.path.join(generated_dir, file))).to(device).repeat(num_centers, 1, 1, 1)
            distances = lpips_fn(img, centers)
            images_centers[idx] = distances.argmin()

    return num_centers, images_files, images_centers.cpu().numpy()


def intra_cluster_dist(num_centers, images_files, images_centers, images_folder, cluster_size=50):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        rng = default_rng()
        avg_dist = torch.zeros([num_centers, ])
        for k in range(num_centers):

            images_idx = np.argwhere(images_centers == k)
            rng.shuffle(images_idx)
            images_idx = images_idx[:cluster_size, :]

            dists = []
            for i in range(len(images_idx)):
                for j in range(i + 1, len(images_idx)):
                    input1_path = os.path.join(images_folder, images_files[i])
                    input2_path = os.path.join(images_folder, images_files[j])

                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.to(device)
                    input_tensor2 = input_tensor2.to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist)
            dists = torch.tensor(dists)
            print("Cluster %d:  Avg. pairwise LPIPS dist: %f/%f" %
                  (k, dists.mean(), dists.std()))
            avg_dist[k] = dists.mean()

        print("Final avg. %f/%f" % (avg_dist[~torch.isnan(avg_dist)].mean(), avg_dist[~torch.isnan(avg_dist)].std()))


def get_close_far_members(args):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        cluster_size = 50
        base_path = os.path.join("cluster_centers", "%s" % (args.dataset), "%s" % (args.baseline))
        avg_dist = torch.zeros([10, ])
        for k in range(10):
            curr_path = os.path.join(base_path, "c%d" % (k))
            files_list = os.listdir(curr_path)
            files_list.remove('center.png')

            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            min_dist, max_dist = 1000.0, 0.0
            min_ind, max_ind = 0, 0
            # center image
            input1_path = os.path.join(curr_path, 'center.png')
            input_image1 = Image.open(input1_path)
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)

            for i in range(len(files_list)):
                input2_path = os.path.join(curr_path, files_list[i])
                input_image2 = Image.open(input2_path)
                input_tensor2 = preprocess(input_image2)
                input_tensor2 = input_tensor2.to(device)
                dist = lpips_fn(input_tensor1, input_tensor2)
                if dist <= min_dist:
                    min_ind = i
                    min_dist = dist
                if dist >= max_dist:
                    max_ind = i
                    max_dist = dist

            print(min_ind, max_ind)
            if len(files_list) > 0:
                # saving the closest member
                path_closest = os.path.join(curr_path, files_list[min_ind])
                new_closest = os.path.join(curr_path, 'closest.png')
                os.system("cp %s %s" % (path_closest, new_closest))

                # saving the farthest member
                path_farthest = os.path.join(curr_path, files_list[max_ind])
                new_farthest = os.path.join(curr_path, 'farthest.png')
                os.system("cp %s %s" % (path_farthest, new_farthest))
            else:
                print("no members in cluster %d" % (k))


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", type=int, default=50)
    parser.add_argument("--cluster_size_generate", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--center_dir", type=str, required=True)
    parser.add_argument("--network_pkl", type=str, required=True)

    args = parser.parse_args()

    n_centers, img_files, img_centers = assign_to_cluster_centers(args.cluster_size_generate, args.outdir,
                                                                          args.center_dir, args.network_pkl,
                                                                          seed=args.seed, batch_size=args.batch_size)
    intra_cluster_dist(n_centers, img_files, img_centers, args.outdir)
