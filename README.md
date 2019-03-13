# B³D compression library

B³D is a fast (~1 GB/s), GPU based image compression method, developed for
fluorescence microscopy applications. Alongside lossless compression, it offers
a noise dependent lossy compression mode, where the loss can be tuned as a
proportion of the inherent image noise (accounting for photon shot noise and
camera read noise). It not only allows for fast compression during imaging, but
can achieve compression ratios up to 100 fold.

Corresponding publication:
[http://www.biorxiv.org/content/early/2017/07/21/164624](http://www.biorxiv.org/content/early/2017/07/21/164624)

## Requirements
* Microsoft Windows 7 or newer, 64-bit
* HDF5 1.8.15 64-bit or later installed. Obtain the
latest version here: [https://support.hdfgroup.org/HDF5/release/obtain518.html](https://support.hdfgroup.org/HDF5/release/obtain518.html)
* Microsoft Visual C++ 2013 Redistributable (x64) (installer included in
  `vcredist` folder)
* CUDA capable graphics card, minimum Compute Capability 3.0
* CUDA 8 compatible graphics driver (for GeForce products 378.66 or later)

HDF5 1.10.x versions are currently **not** supported.

## Installation
All necessary binary files can be found in the `win64` folder. In order for
HDF5 to recognize the filter, the path of this folder has to be stored in the
`%HDF5_PLUGIN_PATH%` environment variable. This can be done automatically by
running the `install_b3d.bat` batch file. It will set the environment variable
for the current user, or if it already exists (e.g. other filters are already
in use), it will copy the necessary files there.

## Reading compressed files
If all the requiriements are present, and installation was successfull, any
HDF5 application should be able to automatically use the filter. This
functionality can be tested by opening the provided example file
`B3D_demo_drosophila_compressed.h5` in HDFView for example.

### Reading files in Fiji
Fiji comes with its own distribution of HDF5 in the form of
`jhdf5-14.12.5.jar` in  the `Fiji.app/jars` folder. On Windows machines this
is **not** compatible with the filter because a [different version of Visual
Studio](http://siomsystems.com/mixing-visual-studio-versions/) was used to
compile the native libraries. To circumvent this issue, we provide a recompiled
version of the same file, included in the `win64` folder. Copy this file to the
`Fiji.app/jars` folder, overwriting the original if necessary.

To open HDF5 files, you need to enable the HDF5 update site (Help > Update,
then click on "Manage update sites" and select "HDF5" from the list). During
the update process you might get a warning message: "There are locally modified
files". As this is a consequence of the previous step, it's important not to
let Fiji overwrite the locally modified files.

## Writing compressed files
To write compressed files, the compression has to be set by calling
`H5Pset_filter` on the dataset creation property list (`dcpl_id`).

```c
H5Pset_filter(dcpl_id, 32016, H5Z_FLAG_OPTIONAL, cd_nelmts, cd_values);
```
where 32016 is the filter id for B³D compression, and `cd_nelmts` is the number
of elements in `cd_values`. `cd_values` is an array comprising of the
following elements:
* cd_values[0] Quantization step*. Relative to sigma (see paper for details;
  default = 0). Set to 0 for lossless compression.
* cd_values[1] Compression mode. 1: Mode1, 2: Mode2 (see paper for details;
  default = 1)
* cd_values[2] Camera conversion parameter* in DN/e- (default=1000)
* cd_values[3] Camera background offset parameter in DN (default = 0)
* cd_values[4] Camera read noise parameter* in e- (default = 0)
* cd_values[5] Tile size, to optimize parallel execution (default=24)

\* Since `cd_values` can only be integers, for floating point parameters the
value*1000 rounded to integers has to be stored in `cd_values`. For lossless
compression all elements can be omitted.

For more information on how to use HDF5 with compression,
[see the official HDF5 website.](https://support.hdfgroup.org/HDF5/faq/compression.html)

### Performance considerations
Filters can only be applied to datasets if chunking is also enabled. For
optimal performance, we recommend choosing a chunk size ~2 MB (1M elements
with 16 bit data).

### Supported data types
* Datasets of 2 or 3 dimensions
* signed or unsigned 8 bit or 16 bit integers

### Using the filter plugin with Anaconda
Anaconda currently (version 5.0.1) ships with version 1.10.1 of the "hdf5"
package. This is currently not supported by the plugin. In most cases running
```
> conda install h5py
```
will install the compatible packages, and will allow Anaconda users to read/write
compressed files. If this is not the case, you should manually install version
1.8.x (x>15) of the "hdf5" package for Anaconda.

## Usage examples
Example scripts are located in the `sample_scripts` folder. Currently this
includes a Python example (tested with Python 2.7.10 and 3.5.3)

## Contact
For any questions / comments about this software, please send an email to [Balint Balazs](mailto:balint.balazs@embl.de).

## B³D Copyright and Software License
B³D source code, binaries and examples are licensed under the MIT license (see
LICENSE file) with the exception of the subfolders
`source/src/B3D_cudaCompress/reduce` and `source/src/cudaCompress/scan`, which
contain code adapted from CUDPP ([http://cudpp.github.io](http://cudpp.github.io)) and come with a separate license (see file
license.txt in the subfolders).

B³D also includes a modified version of cudaCompress ([https://github.com/m0bl0/cudaCompress](https://github.com/m0bl0/cudaCompress)),
also licensed under the MIT license.

The binary distribution in the `win64` folder also includes the redistributable
CUDA runtime library, which comes with it's own license. For details see the
LICENSE file and CUDA_EULA.pdf in that folder. 