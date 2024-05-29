import inspect
import numpy as np
from math import sin
import matplotlib
from matplotlib import pyplot as plt
import time
import datetime
from PIL import Image
import requests

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty")
except:
    pass


def task_2_1():
    # All functions we want to inspect as lambda functions.
    func_collection = [
        lambda t: sin(t),
        lambda t: sin(t) + (3 * sin(2 * t + 1) - 1),
    ]
    # The region where we want to inspect the function in the time domain.
    # Region given in the task: 0 to 4 * pi
    # Since the FFT needs the number of samples to be the some power of 2, we
    # approximate 4 * pi with 16, since this makes it easier.
    region = (0, 16)
    # Sampling rate given in Hz.
    sampling_rate = 32
    for func in func_collection:
        # Calculate the function values in the time domain.
        (t, x) = calculate_samples(func, region, sampling_rate)
        # Calculate the frequencies using the direct DFT and the FFT.
        f = calculate_frequencies(np.size(x), sampling_rate)
        X_direct_dft = direct_dft(x)
        X_fft = fft(x)
        # Reconstruct the original function using the inverse direct DFT and inverse FFT.
        x_direct_dft_inv = direct_dft_inv(X_direct_dft)
        x_fft_inv = fft_inv(X_fft)
        # Plot the results.
        plot(t, x, x_direct_dft_inv, x_fft_inv, f, X_direct_dft, X_fft, func)


def direct_dft_fft_performance_test():
    # Lets compare the performance of the direct DFT and the FFT.
    # Given a function, sample the function with different number of samples
    # and calculate the frequencies using the direct DFT and FFT. Measure the needed time.
    func = lambda t: sin(t) + 4 * sin(3 * t - 1) + 2 * sin(5 * t) + 0.5 * sin(20 * t)
    region = (0, 16)
    sampling_rate_max_power_of_two = 6
    # Result lists.
    number_of_samples = []
    duration_direct_dft = []
    duration_fft = []
    for sampling_rate_power_of_two in range(2, sampling_rate_max_power_of_two + 1):
        sampling_rate = 2**sampling_rate_power_of_two
        (_, x) = calculate_samples(func, region, sampling_rate)
        number_of_samples.append(np.size(x))
        # DFT.
        start_time = datetime.datetime.now()
        _ = direct_dft(x)
        end_time = datetime.datetime.now()
        duration_direct_dft.append((end_time - start_time).microseconds)
        # FFT.
        start_time = datetime.datetime.now()
        _ = fft(x)
        end_time = datetime.datetime.now()
        duration_fft.append((end_time - start_time).microseconds)
    # Print the results.
    plt.figure(figsize=(12, 6))
    plt.plot(
        number_of_samples,
        duration_direct_dft,
        label="duration of the direct DFT, time complexity: O(n^2)",
    )
    plt.plot(
        number_of_samples,
        duration_fft,
        label="duration of the FFT, time complexity: O(n * log(n))",
    )
    plt.title("Performance comparision of the direct DFT and FFT", fontweight="bold")
    plt.xlabel("number of samples")
    plt.ylabel("duration in microseconds")
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()


def dft_on_image():
    url_night_sky = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.wallpapersden.com%2Fimage%2Fdownload%2F4k-starry-sky-stars-milky-way-galaxy_bGttbWeUmZqaraWkpJRqZmetamZn.jpg&f=1&nofb=1&ipt=326b18ddc7ff8b32ebd4ef7918a46a818be42e96b3df8ea0b33a8608fa25e0cb&ipo=images"
    url_diagonal_line = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freepnglogos.com%2Fuploads%2Fline-png%2Fright-horizontal-thick-black-line-0.png&f=1&nofb=1&ipt=765bd421f5ff727b9f32fe6e6615db8bd3400cb9baa60d7a0be96470ff0f6bf5&ipo=images"
    urls = [url_night_sky, url_diagonal_line]
    for url in urls:
        apply_dft_on_image(url)


def apply_dft_on_image(image_url):
    # Read an image from an URL. This image should have the size 512x512.
    img = np.array(Image.open(requests.get(image_url, stream=True).raw), dtype=np.uint8)
    if img.shape == (512, 512, 4):
        # PNG images have four channels, keep only the first three ones.
        print("PNG image detected, drop the alpha channel.")
        img = img[:, :, :3]
    elif img.shape != (512, 512, 3):
        print(
            "The image used in 'apply_dft_on_image' needs to have the size 512x512x3 but has the shape",
            img.shape,
        )
        print(img)
        return
    # Extract the red, green and blue channel from the RGB image.
    # Flatten the channels so that they are no more 512x512 in shape but 262144x1.
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    # Apply the DFT on each channel using the FFT.
    r_dft = fft(r)
    g_dft = fft(g)
    b_dft = fft(b)
    # Normalizing DFT results for visualization (logarithmic transformation).
    # https://stackoverflow.com/questions/38332642/plot-the-2d-fft-of-an-image
    r_dft_mag = 20 * np.log10(np.abs(r_dft))
    g_dft_mag = 20 * np.log10(np.abs(g_dft))
    b_dft_mag = 20 * np.log10(np.abs(b_dft))
    # Create an image from the DFT results by reshaping and combining the channels.
    img_dft = np.stack(
        (
            r_dft_mag.reshape((512, 512)),
            g_dft_mag.reshape((512, 512)),
            b_dft_mag.reshape((512, 512)),
        ),
        axis=-1,
    ).astype(np.uint8)
    # Now reconstruct the image by applying the inverse DFT.
    # We need to round the values, otherwise we get small errors when casting
    # the datatype from float to uint8, since this will always floor the values.
    r_dft_inv = np.round(fft_inv(r_dft))
    g_dft_inv = np.round(fft_inv(g_dft))
    b_dft_inv = np.round(fft_inv(b_dft))
    img_dft_inv = np.stack(
        (
            r_dft_inv.reshape((512, 512)),
            g_dft_inv.reshape((512, 512)),
            b_dft_inv.reshape((512, 512)),
        ),
        axis=-1,
    ).astype(np.uint8)
    # Plot all three images next to eachother.
    fig, (ax_img, ax_dft, ax_dft_inv) = plt.subplots(1, 3)
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("original image")
    ax_dft.imshow(img_dft)
    ax_dft.set_xticks([])
    ax_dft.set_yticks([])
    ax_dft.set_title("DFT image")
    ax_dft_inv.imshow(img_dft_inv)
    ax_dft_inv.set_xticks([])
    ax_dft_inv.set_yticks([])
    ax_dft_inv.set_title("image reconstructed from the DFT image")
    plt.show()
    # Calculate the total difference of all pixels of the original image
    # and the reconstructed image. The sum should be zero, since the images
    # should be equal.
    number_of_not_matching_pixels = np.sum(img != img_dft_inv)
    print(
        "Number of not matching pixels between the orignal image and the reconstructed one:",
        number_of_not_matching_pixels,
    )
    # print("original", img[:10, :10, :])
    # print("reconstructed", img_dft_inv[:10, :10, :])


def calculate_samples(func, region, sampling_rate):
    t_diff = 1 / sampling_rate
    N = int((region[1] - region[0]) * sampling_rate)
    t = np.array([region[0] + t * t_diff for t in range(N)])
    x = np.array([func(_t) for _t in t])
    return (t, x)


def calculate_frequencies(N, sampling_rate):
    # f_k = k * f_s / N
    f = np.array([k * sampling_rate / N for k in range(N)])
    return f


def direct_dft(x):
    # X_k = sum (from n = 0 to N-1) of (x_n * e ^ (−2i * pi * k * n / N))
    N = np.size(x)
    X = np.zeros((N,), dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def direct_dft_inv(X):
    # x_n = 1/N * sum (from k = 0 to N-1) of (X_k * e ^ (2i * pi * k * n / N))
    N = np.size(X)
    x = np.zeros((N,), dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] = x[n] / N
    # The values in x are still complex numbers. We only keep the real part.
    return np.real(x)


def fft(x):
    # If the size of the input array is 0 or 1, return the array itself.
    N = np.size(x)
    if N <= 1:
        return x
    # Recursive calls to fft() for even and odd indices of the input array.
    even = fft(x[0::2])
    odd = fft(x[1::2])
    # Compute the twiddle factors.
    # Twiddle factors are complex numbers that represent phase shifts for each frequency bin.
    # They are precalculated and multiplied with the FFT results of odd-indexed elements.
    T = np.array([np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)])
    # Combine the results of even and odd FFT computations using butterfly operations.
    # This step calculates the final FFT result.
    # The FFT result for the first half of the array is combined / added with twiddle factors,
    # while the FFT result for the second half is subtracted.
    return np.array(
        [even[k] + T[k] for k in range(N // 2)]
        + [even[k] - T[k] for k in range(N // 2)]
    )


def fft_inv(X):
    N = np.size(X)
    # Apply FFT on the conjugate of the input signal to perform inverse FFT
    X_conj = np.conjugate(X)
    x = fft(X_conj)
    # Scale each element in the inverse FFT result by 1/N to obtain the final inverse FFT result.
    return np.real(np.divide(x, N))


def plot(t, x, x_direct_dft_inv, x_fft_inv, f, X_direct_dft, X_fft, func):
    # We need the functions name for the title of the plot.
    # Therefore extract it from the lambda function.
    lambda_function = inspect.getsource(func)
    func_name = lambda_function.split(":", 1)[1].strip()
    # We want to plot the things:
    # 1. the original sampled function in the time domain.
    # 2. the frequencies obtained by the DFT.
    # 3. the frequencies obtained by the FFT.
    # 4. the reconstructed function (using the inverse direct DFT) in the time domain.
    # 5. the reconstructed function (using the inverse FFT) in the time domain.
    # Plot 1., 4. and 5. within one plot, so we can compare it, as well as 2. and 3.
    fig, (ax_time_domain, ax_frequency_domain) = plt.subplots(2, 1)
    # Time domain.
    ax_time_domain.plot(t, x, "-", label="x")
    ax_time_domain.plot(
        t,
        x_direct_dft_inv,
        "-.",
        label="x (reconstructed using the inverse direct DFT)",
    )
    ax_time_domain.plot(
        t,
        x_fft_inv,
        ":",
        label="x (reconstructed using the inverse FFT)",
    )
    ax_time_domain.set_xlim([min(t), max(t)])
    ax_time_domain.set_title(
        "'" + func_name + "' in the time domain", fontweight="bold"
    )
    ax_time_domain.set_xlabel("Time in s")
    ax_time_domain.set_ylabel("x")
    ax_time_domain.grid(alpha=0.5)
    ax_time_domain.legend()
    # Frequency domain.
    N = np.size(f)
    f = f[: N // 2]
    X_direct_dft = X_direct_dft[: N // 2]
    X_fft = X_fft[: N // 2]
    ax_frequency_domain.plot(
        f, np.abs(X_direct_dft), label="X (obtained using the direct DFT)"
    )
    ax_frequency_domain.plot(f, np.abs(X_fft), "-.", label="X (obtained using the FFT)")
    ax_frequency_domain.set_xlim([min(f), max(f)])
    ax_frequency_domain.set_title(
        "'" + func_name + "' in the frequency domain", fontweight="bold"
    )
    ax_frequency_domain.set_xlabel("Frequency in Hz")
    ax_frequency_domain.set_ylabel("X")
    ax_frequency_domain.grid(alpha=0.5)
    ax_frequency_domain.legend()
    plt.show()


if __name__ == "__main__":
    task_2_1()
    # direct_dft_fft_performance_test()
    # dft_on_image()
