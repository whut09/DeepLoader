import os
from multiprocessing import cpu_count, Pool
from pathlib import Path
from pathlib import PosixPath
from typing import Callable, Dict, List
from typing import Optional
from typing import Union, Tuple

import numpy as np
import tqdm
from PIL import Image
from deeploader.util.fileutil import get_file_size

IMG_FORMATS = ['JPG', 'GIF', 'JPEG', 'PNG', 'BMP']


def preprocess_image(
        image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.

    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.

    Returns:
        A numpy array of the processed image.
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return np.array(image_pil).astype('uint8')


def load_image(
        image_file: Union[PosixPath, str],
        target_size: Tuple[int, int] = None,
        grayscale: bool = False,
        img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.

    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
    """
    try:
        img = Image.open(image_file)

        # validate image format
        if img.format not in img_formats:
            print('Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            img = preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return img

    except Exception as e:
        print('Invalid image file {image_file}:\n{e}')
        return None


def parallelise(function: Callable, data: List) -> List:
    pool = Pool(processes=cpu_count())
    results = list(tqdm.tqdm(pool.imap(function, data), total=len(data)))
    pool.close()
    pool.join()
    return results


class Hashing:
    """
    Find duplicates using hashing algorithms and/or generate hashes given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encoding generation:
    To generate hashes using specific hashing method. The generated hashes can be used at a later time for
    deduplication. Using the method 'encode_image' from the specific hashing method object, the hash for a
    single image can be obtained while the 'encode_images' method can be used to get hashes for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self) -> None:
        self.target_size = (8, 8)  # resizing to dims

    def get_image_feat(
            self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            hash: A 16 character hexadecimal string hash for the image.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        myhash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        myhash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            # print(image_file)
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')
        if isinstance(image_pp, np.ndarray):
            hash_mat = self._hash_algo(image_pp)
            feat = np.float32(hash_mat.flatten())
            feat[0:2] = 0
            return feat
        else:
            return None

    def get_image_feats(self, image_dir=None):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        image_dir = Path(image_dir)

        files = [
            i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

        print('Start: Calculating hashes...')
        raw_files = [str(f) for f in files]
        # sort files by size descent
        sizes = np.array([get_file_size(f) for f in raw_files])
        indexs = np.argsort(sizes)[::-1]
        _files = []
        for i in range(len(raw_files)):
            _files.append(raw_files[indexs[i]])

        hashes = parallelise(self.get_image_feat, _files)
        images = []
        feats = []
        for idx, hash in enumerate(hashes):
            if hash is None:
                continue
            images.append(_files[idx])
            feats.append(hash)
        print('End: Calculating hashes!')
        return images, np.stack(feats)

    def _hash_algo(self, image_array: np.ndarray):
        pass

    def _hash_func(self, image_array: np.ndarray):
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)


class PHash(Hashing):
    """
    Inherits from Hashing base class and implements perceptual hashing (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Perceptual hash for images
    from imagededup.methods import PHash
    phasher = PHash()
    perceptual_hash = phasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    perceptual_hash = phasher.encode_image(image_array = <numpy image array>)
    OR
    perceptual_hashes = phasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import PHash
    phasher = PHash()
    duplicates = phasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = phasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import PHash
    phasher = PHash()
    files_to_remove = phasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = phasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the perceptual hash of the image.
        """
        from scipy.fftpack import dct
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
                           : self.__coefficient_extract[0], : self.__coefficient_extract[1]
                           ]

        # median of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class AHash(Hashing):
    """
    Inherits from Hashing base class and implements average hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Average hash for images
    from imagededup.methods import AHash
    ahasher = AHash()
    average_hash = ahasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    average_hash = ahasher.encode_image(image_array = <numpy image array>)
    OR
    average_hashes = ahasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import AHash
    ahasher = AHash()
    duplicates = ahasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = ahasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import AHash
    ahasher = AHash()
    files_to_remove = ahasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = ahasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (8, 8)

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        hash_mat = image_array >= avg_val
        return hash_mat


class MHash(Hashing):
    """
    Inherits from Hashing base class and implements average hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Average hash for images
    from imagededup.methods import AHash
    ahasher = AHash()
    average_hash = ahasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    average_hash = ahasher.encode_image(image_array = <numpy image array>)
    OR
    average_hashes = ahasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import AHash
    ahasher = AHash()
    duplicates = ahasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = ahasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import AHash
    ahasher = AHash()
    files_to_remove = ahasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = ahasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (8, 8)

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        return image_array / (avg_val + 0.01)


class DHash(Hashing):
    """
    Inherits from Hashing base class and implements difference hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Difference hash for images
    from imagededup.methods import DHash
    dhasher = DHash()
    difference_hash = dhasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    difference_hash = dhasher.encode_image(image_array = <numpy image array>)
    OR
    difference_hashes = dhasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import DHash
    dhasher = DHash()
    duplicates = dhasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = dhasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import DHash
    dhasher = DHash()
    files_to_remove = dhasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = dhasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (9, 8)

    def _hash_algo(self, image_array):
        """
        Get difference hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the difference hash of the image.
        """
        # Calculates difference between consecutive columns and return mask
        hash_mat = image_array[:, 1:] > image_array[:, :-1]
        return hash_mat


class WHash(Hashing):
    """
    Inherits from Hashing base class and implements wavelet hashing. (Implementation reference:
    https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Wavelet hash for images
    from imagededup.methods import WHash
    whasher = WHash()
    wavelet_hash = whasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    wavelet_hash = whasher.encode_image(image_array = <numpy image array>)
    OR
    wavelet_hashes = whasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import WHash
    whasher = WHash()
    duplicates = whasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = whasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import WHash
    whasher = WHash()
    files_to_remove = whasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = whasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (256, 256)
        self.__wavelet_func = 'haar'

    def _hash_algo(self, image_array):
        """
        Get wavelet hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the wavelet hash of the image.
        """
        # decomposition level set to 5 to get 8 by 8 hash matrix
        import pywt
        image_array = image_array / 255
        coeffs = pywt.wavedec2(data=image_array, wavelet=self.__wavelet_func, level=5)
        LL_coeff = coeffs[0]

        # median of LL coefficients
        median_coef_val = np.median(np.ndarray.flatten(LL_coeff))

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = LL_coeff >= median_coef_val
        return hash_mat
