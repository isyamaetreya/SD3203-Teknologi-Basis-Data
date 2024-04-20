---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="eBBAmPejJEFJ"}
# **Three Ways of Storing and Accessing Lots of Images in Python**
:::

::: {.cell .markdown id="H0M9nrzSJHVu"}
Berdapat tiga cara utama untuk menyimpan dan mengakses gambar dengan
Python, yaitu menyimpan gambar sebagai file .pgn, menyimpan gambar dalam
database LMDB, dan menyimpan gambar dalam format HDFS.

Pemilihan motode penyimpanan yang tepat dapat dilakukan berdasarkan
tugas dan ukuran dari dataset. Kinerja saat membaca dan menulis gambar
tunggal berbeda. Hal ini dapat disebabkan dari penggunaan metode yang
berbeda dan banyaknya jumlah gambar.

Berikut ini adalah tutorial untuk cara menyimpan dan mengelola gambar
dengan Python.
:::

::: {.cell .markdown id="sKOWrZyZLBA0"}
## **Import Dataset**
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="kJQptHilLDpK" outputId="b1381a65-c475-49ea-d810-b39ace705914"}
``` python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("/content/sample_data/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

::: {.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)
:::
:::

::: {.cell .markdown id="9Qv9UZheLN2x"}
Dataset yang digunakan adalah CIFAR-10. Dataset ini sering digunakan
untuk melatih model machine learning untuk mengenali objek dalam gambar.
Pada dataset terdapat 60 ribu gambar dengan ukuran 32x32 piksel dan
terdapat 10 kelas.
:::

::: {.cell .markdown id="avx4GVySMO1b"}
## **Setup for Storing Images on Disk**
:::

::: {.cell .markdown id="q5J1NukGLtlr"}
Install library pillow, untuk mengolah gambar dari berbagai jenis format
file gambar, misalnya PNG, GIF, JPG, dan lain-lain.
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="HzpG9BboLo2a" outputId="45325dbb-b8dd-4dae-f150-a8a090dfedd5"}
``` python
!pip install pillow
```

::: {.output .stream .stdout}
    Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (9.4.0)
:::
:::

::: {.cell .markdown id="URJJIoCAMUHN"}
## **Getting Started With LMDB**
:::

::: {.cell .markdown id="Mgn7W_r7MZeC"}
Install library lmdb yang akan digunakan untuk menyimpan dan mengakses
data secara cepat.
:::

::: {.cell .code execution_count="14" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="jAMiRihCMX_0" outputId="6d1a7094-7f19-4152-d942-1b3b5b595fbd"}
``` python
!pip install lmdb
```

::: {.output .stream .stdout}
    Collecting lmdb
      Downloading lmdb-1.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (299 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 299.2/299.2 kB 4.6 MB/s eta 0:00:00
    db
    Successfully installed lmdb-1.4.1
:::
:::

::: {.cell .markdown id="e1ysWTV0MlRL"}
## **Getting Started With HDF5**
:::

::: {.cell .markdown id="dA5ICS9QMqww"}
Install library H5PY yang digunakan untuk mengolah data pada dataset
yang berformart HDF5.
:::

::: {.cell .code execution_count="15" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="CxDeUuevMoGg" outputId="d7ab548b-bfe5-4c1d-f6ca-cb378b0f5a24"}
``` python
!pip install h5py
```

::: {.output .stream .stdout}
    Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (3.9.0)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.10/dist-packages (from h5py) (1.25.2)
:::
:::

::: {.cell .markdown id="y8JU5JfHKtXW"}
# **Storing a Single Image**
:::

::: {.cell .markdown id="CXJ4g3FBOWKA"}
Pada bagian ini, akan membandingkan waktu yang dibutuhkan untuk membaca
dan menyimpan file dan banyak memori disk yang dgunakan dari
masing-masing metode cara penyimpanan satu gambar.

Langkah yang pertama adalah membuat folder untuk setiap metode, dengan
kode berikut:
:::

::: {.cell .code execution_count="16" id="pN4AW1aZNE6m"}
``` python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```
:::

::: {.cell .markdown id="CD-YN8PSPfMe"}
atau dengan kode berikut:
:::

::: {.cell .code execution_count="17" id="L9pW8z6QNHLj"}
``` python
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
:::

::: {.cell .markdown id="ejNfb_DYK0ZL"}
## **1. Storing to Disk** {#1-storing-to-disk}
:::

::: {.cell .markdown id="35A9_GBsP7rA"}
Pada kode di bawah ini, digunakan untuk menyimpan gambar dalam format
.png dengan munggunakan PIL dan menyimpan label gambar dengan format
csv. Hal ini akan membuat pengelolaan informasi gambar menjadi lebih
sederhana dan mudah diakses.
:::

::: {.cell .code execution_count="18" id="ISyyrFBMNM1s"}
``` python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```
:::

::: {.cell .markdown id="IJfy4Xx9NP9s"}
## **2. Storing to LMDB** {#2-storing-to-lmdb}
:::

::: {.cell .markdown id="Gu2R6WNzQjBF"}
Dengan menggunakan kelas CIFAR berguna untuk mewakili satu gambar dari
dataset. kelas CIFAR akan menyimpan representasi dari satu gambar yang
ada pada dataset. Saat objek dibuat, gambardiubah menjadi byte array
untuk disimpan bersama dengan labelnya.
:::

::: {.cell .code execution_count="19" id="k9mZkLu0NUv7"}
``` python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```
:::

::: {.cell .markdown id="nT-uW1hvRMnN"}
Modul lmdb berfungsi untuk berkerja dengan database LMBD dan modul
pickle berfungsi untuk menyimpan satu gambar ke dalam database LMBD.

Fungsi store_single_lmd Fungsi ini digunakan untuk menyimpan satu gambar
beserta labelnya ke dalam database LMDB untuk digunakan dalam pengolahan
data atau pembelajaran mesin.
:::

::: {.cell .code execution_count="20" id="181vtgYwNWbZ"}
``` python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 1024

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```
:::

::: {.cell .markdown id="bhS_kcLONZOW"}
## **3. Storing With HDF5** {#3-storing-with-hdf5}
:::

::: {.cell .markdown id="sn9TqK44SFW2"}
Fungsi store_single_hdf5 ini digunakan untuk menyimpan satu gambar ke
dalam file HDF5. Ini memanfaatkan modul h5py yang merupakan modul Python
untuk bekerja dengan file HDF5.

Ini adalah cara sederhana dan efisien untuk menyimpan informasi gambar
ke dalam file HDF5, yang kemudian dapat digunakan untuk pembelajaran
mesin atau analisis data lainnya.
:::

::: {.cell .code execution_count="21" id="JBbD3ogdNYqC"}
``` python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```
:::

::: {.cell .markdown id="TWG7WFDuNkQ8"}
## **4. Experiments for Storing a Single Image** {#4-experiments-for-storing-a-single-image}
:::

::: {.cell .markdown id="JOiH4nKWS4hi"}
Kode berikut merupakan dictionary bernama \_store_single_funcs yang
berisi fungsi-fungsi untuk menyimpan gambar dalam format yang berbeda,
yaitu \"disk\", \"lmdb\", dan \"hdf5\". Setiap kunci dalam dictionary
ini adalah string yang merepresentasikan jenis penyimpanan, dan nilainya
adalah fungsi-fungsi yang sesuai untuk melakukan penyimpanan ke jenis
penyimpanan tersebut.
:::

::: {.cell .code execution_count="22" id="gUqoETguNpUe"}
``` python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```
:::

::: {.cell .markdown id="lcEbwVJNTT6O"}
Kode berikut digunakan untuk dapat melihat dan membandingkan waktu yang
dibutuhkan untuk menyimpan gambar menggunakan berbagai metode
penyimpanan
:::

::: {.cell .code execution_count="23" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2dCc0XWjNq4S" outputId="90cdb7af-60e3-4043-96d0-3e19955557b2"}
``` python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.02989574800039918
    Method: lmdb, Time usage: 0.013732802999584237
    Method: hdf5, Time usage: 0.005884657000024163
:::
:::

::: {.cell .markdown id="bdXUb8rfTW-t"}
Hasil berdasarkan kode di atas adalah,

1.  Disk 2.98 milisecond
2.  LMDB 1.37 milisecond
3.  HDF5 0.588 milisecond Semua metode sangat cepat untuk menyimpan
    gambar.
:::

::: {.cell .markdown id="OZL7ZRcwT7qO"}
# **Storing Many Images**
:::

::: {.cell .markdown id="OuT_C3saWvCQ"}
Berikut ini adalah kode untuk menyimpan banyak gambar dan kemudian
menjalankan eksperimen berjangka waktu.
:::

::: {.cell .markdown id="jVkm6O4gUmbC"}
## **1. Adjusting the Code for Many Images** {#1-adjusting-the-code-for-many-images}
:::

::: {.cell .markdown id="WsmhnJ4AW-NZ"}
Berikut adalah tiga fungsi baru untuk menyimpan banyak gambar ke dalam
berbagai format penyimpanan: \"disk\", \"lmdb\", dan \"hdf5\". Dengan
menggunakan ketiga fungsi ini, dapat dengan mudah menyimpan banyak
gambar sekaligus ke dalam format yang berbeda-beda, sesuai dengan
kebutuhan dan preferensi Anda dalam proses pengolahan dan penyimpanan
data.
:::

::: {.cell .code execution_count="26" id="IrD5Pmt2TgAB"}
``` python
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```
:::

::: {.cell .markdown id="wiv3Cg9gU5NT"}
## **2. Preparing the Dataset** {#2-preparing-the-dataset}
:::

::: {.cell .markdown id="Z3eKp3dzXPWq"}
Pada dataset yang akan dibuat, memiliki 100.000 gambar dan 100.000 label
yang siap untuk diukur waktu penyimpanannya ke dalam berbagai format
penyimpanan seperti \"disk\", \"lmdb\", dan \"hdf5\". Perintah
print(np.shape(images)) dan print(np.shape(labels)) digunakan untuk
memastikan bahwa dimensi dari images dan labels sekarang adalah (100000,
32, 32, 3) dan (100000, 1) untuk memastikan jumlah sesuai yang
diinginkan.
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ex2gdYLLTfht" outputId="bf30220e-e729-4ad6-8f7a-aec18b50ce0e"}
``` python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```

::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)
:::
:::

::: {.cell .markdown id="9IOv2aXKVCyx"}
## **3. Experiment for Storing Many Images** {#3-experiment-for-storing-many-images}
:::

::: {.cell .markdown id="XR6iPgMMX674"}
Berikut adalah kode untuk mengukur waktu secara iteratif yang dibutuhkan
untuk menyimpan sejumlah gambar (sesuai dengan cutoff) ke dalam tiga
format penyimpanan yang berbeda. Ini dilakukan untuk mengukur efisiensi
dan performa dari masing-masing metode penyimpanan. Outputnya adalah
waktu eksekusi untuk setiap metode dan cutoff yang diukur.
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="W0_aMvqDVFlK" outputId="c7a2d4a0-18be-43db-d081-b673328326bf"}
``` python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.012243940000189468
    Method: lmdb, Time usage: 0.012275134000446997
    Method: hdf5, Time usage: 0.005583809000199835
    Method: disk, Time usage: 0.06700411000019812
    Method: lmdb, Time usage: 0.025000290000207315
    Method: hdf5, Time usage: 0.002822978000040166
    Method: disk, Time usage: 0.5644104540006083
    Method: lmdb, Time usage: 0.08099333300015132
    Method: hdf5, Time usage: 0.011943067000174779
    Method: disk, Time usage: 4.530555383000319
    Method: lmdb, Time usage: 0.5065041340003518
    Method: hdf5, Time usage: 0.06901065399961226
    Method: disk, Time usage: 63.80812274900018
    Method: lmdb, Time usage: 4.411114966000241
    Method: hdf5, Time usage: 0.8583474789993488
:::
:::

::: {.cell .markdown id="EqGd7PzuYFNd"}
Fungsi plot_with_legend digunakan untuk membuat plot dengan beberapa
dataset yang memiliki legenda yang sesuai. Output dari kode di bawah ini
adalah akan menampilkan perbandingan waktu penyimpanan antara tiga
metode (\"disk\", \"lmdb\", \"hdf5\") terhadap jumlah gambar yang
disimpan. Jika ingin melihat perbedaan skala logaritmik, dapat dilihat
pada plot kedua (\"Log storage time\").
:::

::: {.cell .code execution_count="29" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="jTyzD4lhVM83" outputId="00d51470-3c60-44df-a42c-b286d0f37b47"}
``` python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

::: {.output .stream .stderr}
    <ipython-input-29-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_1a38dc3d5b4b4fbd807f5fb381773d16/4b2ac2f727f642d3fc47776fa18d37581ce194c6.png)
:::

::: {.output .stream .stderr}
    <ipython-input-29-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_1a38dc3d5b4b4fbd807f5fb381773d16/758b086fc1f01a4ea97fc9cf005852df80647571.png)
:::
:::

::: {.cell .markdown id="A9Fx86glYfbF"}
Berdasarkan Plot pertama: menunjukkan waktu penyimpanan normal yang
tidak disesuaikan, menyoroti perbedaan drastis antara penyimpanan ke
file dan LMDB atau HDF5, .png

Berdasarkan Plot kedua: menunjukkan pengaturan waktu, menyoroti bahwa
HDF5 dimulai lebih lambat dari LMDB tetapi, dengan jumlah gambar yang
lebih besar, keluar sedikit di depan.log
:::

::: {.cell .markdown id="AytXtkX2ZJw-"}
# **Reading a Single Image**
:::

::: {.cell .markdown id="DwARmtXBZRgi"}
## **1. Reading From Disk** {#1-reading-from-disk}
:::

::: {.cell .markdown id="2CDByDmsaDcI"}
Fungsi \"read_single_disk\" digunakan untuk membaca satu gambar dan
label terkait dari disk dengan menggunakan ID unik image_id.
:::

::: {.cell .code execution_count="31" id="ds22KRzzZszT"}
``` python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```
:::

::: {.cell .markdown id="vjV_5zRMZYtK"}
## **2. Reading From LMDB** {#2-reading-from-lmdb}
:::

::: {.cell .markdown id="Z80Q_nudaMXH"}
Fungsi \"read_single_lmdb\" digunakan untuk membaca satu gambar dan
label terkait dari LMDB dengan menggunakan ID unik image_id.
:::

::: {.cell .code execution_count="32" id="dATPA_-RZvv7"}
``` python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```
:::

::: {.cell .markdown id="ulAiJj3cZdDk"}
## **3. Reading From HDF5** {#3-reading-from-hdf5}
:::

::: {.cell .markdown id="RvuoYNwvaasI"}
Fungsi \"read_single_hdf5\" digunakan untuk membaca satu gambar dan
label terkait dari file HDF5 dengan menggunakan ID unik image_id.
:::

::: {.cell .code execution_count="33" id="HDwA5m9zZxhq"}
``` python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```
:::

::: {.cell .code execution_count="34" id="iuFytnwOZz7P"}
``` python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```
:::

::: {.cell .markdown id="HnA5bZksZjzG"}
## **4. Experiment for Reading a Single Image** {#4-experiment-for-reading-a-single-image}
:::

::: {.cell .markdown id="vZSPeDYWaslG"}
Kode berikut digunakan untuk mengukur waktu yang dibutuhkan dalam
membaca satu gambar dan label terkait dari metode penyimpanan yang
berbeda.

Pengukuran waktu ini membantu membandingkan efisiensi dari masing-masing
metode dalam membaca satu gambar dan label terkait. Semakin kecil waktu
yang dibutuhkan, semakin efisien metode tersebut dalam operasi membaca.
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8VxDS4YlZ1kz" outputId="a0136104-99a1-4fb2-de1b-2a2a94b64349"}
``` python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.011441632000241952
    Method: lmdb, Time usage: 0.011842079999951238
    Method: hdf5, Time usage: 0.008050738999372697
:::
:::

::: {.cell .markdown id="Crrqgh8Va1qO"}
Berdasarkan output di atas, metode HDF5 memiliki waktu tercapat dalam
membaca satu gambar dan labelnya.
:::

::: {.cell .markdown id="S116I_fLbAp2"}
# **Reading Many Images**
:::

::: {.cell .markdown id="DuP6upm-bGy5"}
## **1. Adjusting the Code for Many Images** {#1-adjusting-the-code-for-many-images}
:::

::: {.cell .markdown id="2fhQaOM0b_YZ"}
Fungsi-fungsi di bawah ini digunakan untuk membaca banyak gambar dari
berbagai metode penyimpanan.
:::

::: {.cell .code execution_count="36" id="XbhZXqpvbiCy"}
``` python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```
:::

::: {.cell .markdown id="Ku_uCO5xbLGg"}
## **2. Experiment for Reading Many Images** {#2-experiment-for-reading-many-images}
:::

::: {.cell .markdown id="oWBesmyPcaJG"}
Kode di bawah ini digunakan untuk mengukur waktu yang diperlukan dalam
membaca sejumlah gambar dari berbagai metode penyimpanan: disk, LMDB,
dan HDF5.

Setelah setiap iterasi, hasil pengukuran waktu akan dicetak. Hasilnya
adalah waktu yang diperlukan untuk membaca jumlah gambar tertentu dari
setiap metode penyimpanan.

Pengukuran waktu ini memberikan informasi tentang kecepatan membaca
gambar dari disk, LMDB, dan HDF5 untuk jumlah gambar tertentu.
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7iKVsK0UbbIv" outputId="c74e9270-ffb1-412c-f67b-89dfb2bceb86"}
``` python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, No. images: 10, Time usage: 0.009754140999575611
    Method: lmdb, No. images: 10, Time usage: 0.0030604690000473056
    Method: hdf5, No. images: 10, Time usage: 0.0038256640000327025
    Method: disk, No. images: 100, Time usage: 0.04793030600012571
    Method: lmdb, No. images: 100, Time usage: 0.009664358000009088
    Method: hdf5, No. images: 100, Time usage: 0.01156530499974906
    Method: disk, No. images: 1000, Time usage: 0.42482068200024514
    Method: lmdb, No. images: 1000, Time usage: 0.03501148800023657
    Method: hdf5, No. images: 1000, Time usage: 0.03544359999978042
    Method: disk, No. images: 10000, Time usage: 4.3210263660002965
    Method: lmdb, No. images: 10000, Time usage: 0.29629048999959195
    Method: hdf5, No. images: 10000, Time usage: 0.2852194690003671
    Method: disk, No. images: 100000, Time usage: 44.21770895499958
    Method: lmdb, No. images: 100000, Time usage: 2.245065451999835
    Method: hdf5, No. images: 100000, Time usage: 1.0041151519999403
:::
:::

::: {.cell .markdown id="AiEVHSW_c2fO"}
Kode di bawah ini digunakan untuk membuat plot grafik yang menampilkan
hasil pengukuran waktu untuk membaca sejumlah gambar dari disk, LMDB,
dan HDF5.

Grafik pertama menampilkan waktu membaca dalam skala linier. Sedangkan
grafik kedua menampilkan waktu membaca dalam skala logaritmik,
memberikan pemahaman lebih baik tentang perbandingan waktu di antara
metode penyimpanan.
:::

::: {.cell .code execution_count="38" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="hxun4r2mbow9" outputId="1a04cb7b-89a8-41ad-b4f3-b7519cf95380"}
``` python
disk_x_r = read_many_timings["disk"]
lmdb_x_r = read_many_timings["lmdb"]
hdf5_x_r = read_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Read time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Log read time",
    log=True,
)
```

::: {.output .stream .stderr}
    <ipython-input-30-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_1a38dc3d5b4b4fbd807f5fb381773d16/fd7576ff5c1566abfd2f0621b82e133447020785.png)
:::

::: {.output .stream .stderr}
    <ipython-input-30-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_1a38dc3d5b4b4fbd807f5fb381773d16/9baa11d25a5c265d0e97e00698e3de4c188c7ecc.png)
:::
:::

::: {.cell .markdown id="WfD8M_uXdHoK"}
Grafik di atas menunjukkan waktu baca normal yang tidak disesuaikan,
menunjukkan perbedaan drastis antara pembacaan dari file dan LMDB atau
HDF5. .png

Grafik di bawah ini menunjukkan pengaturan waktu, menyoroti perbedaan
relatif dengan gambar yang lebih sedikit. Yaitu, kita dapat melihat
bagaimana HDF5 dimulai di belakang tetapi, dengan lebih banyak gambar,
menjadi lebih cepat secara konsisten daripada LMDB dengan margin
kecil.log
:::

::: {.cell .code execution_count="39" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":603}" id="tzIgrh9nbv4H" outputId="774bb252-b51a-4b28-9246-a6e065c942c3"}
``` python
plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r, disk_x, lmdb_x, hdf5_x],
    [
        "Read PNG",
        "Read LMDB",
        "Read HDF5",
        "Write PNG",
        "Write LMDB",
        "Write HDF5",
    ],
    "Number of images",
    "Seconds",
    "Log Store and Read Times",
    log=False,
)
```

::: {.output .stream .stderr}
    <ipython-input-30-99d89538a067>:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_1a38dc3d5b4b4fbd807f5fb381773d16/a2f85858d76aaac509630126b828a63e6f3dbe00.png)
:::
:::

::: {.cell .markdown id="3pUzsOZkeILX"}
# **Kesimpulan**
:::

::: {.cell .markdown id="yG9qzYYgeK6S"}
1.  Disk (PNG) adalah pilihan yang paling sederhana tetapi memakan ruang
    disk yang lebih besar dan memerlukan waktu lebih lama untuk membaca
    dan menulis pada skala besar.
2.  LMDB memberikan keseimbangan yang baik antara efisiensi ruang disk
    dan kinerja, sangat cocok untuk jumlah gambar besar dalam
    pengembangan aplikasi.
3.  HDF5 cocok untuk penggunaan yang memerlukan hierarki data yang
    kompleks dan kinerja yang baik saat mengakses data dalam jumlah
    besar.

Pilihan metode penyimpanan tergantung pada kebutuhan aplikasi dan
prioritas dalam hal kinerja, efisiensi disk, dan kompleksitas
implementasi. Dengan memahami kelebihan dan kekurangan masing-masing,
kita dapat memilih yang paling sesuai untuk kasus penggunaan kita.
:::
