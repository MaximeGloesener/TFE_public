import tensorrt as trt
import pycuda.driver as cuda
import os 
import torch 

class Calibrator(trt.IInt8Calibrator):
    def __init__(self, training_data, cache_file, batch_size=64, algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        """
        Parameters
        ----------
        training_data : numpy array
            The data using to calibrate quantization model
        cache_file : str
            The path user want to store calibrate cache file
        batch_size : int
            The batch_size of calibrating process
        algorithm : tensorrt.tensorrt.CalibrationAlgoType
            The algorithms of calibrating contains LEGACY_CALIBRATION,
            ENTROPY_CALIBRATION, ENTROPY_CALIBRATION_2, MINMAX_CALIBRATION.
            Please refer to https://docs.nvidia.com/deeplearning/tensorrt/api/
            python_api/infer/Int8/Calibrator.html for detail
        """
        trt.IInt8Calibrator.__init__(self)

        self.algorithm = algorithm
        self.cache_file = cache_file

        self.dataloader = training_data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.dataloader.dataset[0][0].numpy().nbytes * self.batch_size)

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        This function is used to define the way of feeding calibrating data each batch.

        Parameters
        ----------
        names : str
             The names of the network inputs for each object in the bindings array

        Returns
        -------
        list
            A list of device memory pointers set to the memory containing each network
            input data, or an empty list if there are no more batches for calibration.
            You can allocate these device buffers with pycuda, for example, and then
            cast them to int to retrieve the pointer
        """
        # self.data is a pytorch dataloader, iterate over it to get the data as numpy arrays for tensorrt
        if self.current_index + self.batch_size > len(self.dataloader.dataset):
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print(f"Calibrating batch {current_batch}, containing {self.batch_size} images")

        batch = []
        for i in range(self.current_index, self.current_index + self.batch_size):
            image = self.dataloader.dataset[i][0]  # Assuming the dataset returns (image, label)
            batch.append(image.numpy().ravel())
        batch = torch.tensor(batch).flatten().numpy()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        memory_pointers = [self.device_input]
        return memory_pointers

    def read_calibration_cache(self):
        """
        If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.

        Returns
        -------
        cache object
            A cache object which contains calibration parameters for quantization
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Write calibration cache to specific path.

        Parameters
        ----------
        cache : str
             The calibration cache to write
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)