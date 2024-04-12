from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor
import time
import os
import glob
import torch
from multiprocessing import get_start_method
from pytorch_lightning.plugins import TorchCheckpointIO
 
from nemo.utils import logging
from nemo.utils.s3_utils import S3Utils
 
from nemo.utils.checkpoint_file_utils import parse_prefix_with_step
 
 
_PATH = Union[str, Path]
SHARED_MEM_DIR = '/dev/shm'
DEFAULT_CHUNK_SIZE_MB = 64
DEFAULT_MAX_READ_CONCURRENCY = 15
DEFAULT_MAX_WRITE_CONCURRENCY = 10
 
class S3CheckpointIO(TorchCheckpointIO):
    """A custom S3CheckpointIO module that supports checkpoint reading/writing with s3 when filepath
    is a s3 url, otherwise default to TorchCheckpointIO.
    """
 
    def __init__(self, chunk_size_MB=DEFAULT_CHUNK_SIZE_MB, max_read_concurrency=DEFAULT_MAX_READ_CONCURRENCY, max_write_concurrency=DEFAULT_MAX_WRITE_CONCURRENCY):
        """
        Initialize the transfer configuration with custom values.
 
        This method overrides the default TransferConfig values in boto3.
        See https://boto3.amazonaws.com/v1/documentation/api/latest/_modules/boto3/s3/transfer.html#TransferConfig
 
        Args:
            chunk_size_MB (int, optional): The size of chunks to use when transferring files.
                Default is 64 (MB).
            max_read_concurrency (int, optional): The maximum number of threads that will be making
                requests to perform a download. Default is 15.
            max_write_concurrency (int, optional): The maximum number of threads that will be making
                requests to perform an upload. Default is 10.
        """
        self.chunk_size_MB = chunk_size_MB
        self.max_read_concurrency = max_read_concurrency 
        self.max_write_concurrency = max_write_concurrency 
        self._clean_up_temp_files()
        super().__init__()
 
 
    def _clean_up_temp_files(self):
        """
        Cleans up the files in the shared memory directory. 
        """
        file_pattern = os.path.join(SHARED_MEM_DIR, 'tmp*')
        for filename in glob.glob(file_pattern):
            try:
                os.remove(filename)
            except Exception as e:
                logging.info(f"Error occurred while deleting file {filename}: {e}")
 
 
    def _serialize_checkpoint_to_shm(self, checkpoint: Dict, path: str) -> str:
        """
        Seralizes the checkpoint to shared memory format. 
 
        Returns:
            filename of the temporary file in shared memory.
        """
        ss = time.perf_counter()
        tempfile = NamedTemporaryFile(dir=SHARED_MEM_DIR, delete=False)
        torch.save(checkpoint, tempfile)
        tt = time.perf_counter() - ss
        logging.info(f'Time elapsed saving checkpoint dict to {tempfile.name} for {path}: {tt:.2f} seconds, rank {torch.distributed.get_rank()}')
        del checkpoint
        return tempfile.name
    
 
    def _serialize_checkpoint_to_bytes(self, checkpoint: Dict, path: str) -> BytesIO:
        """
        Seralizes the checkpoint to bytes. 
 
        Returns:
            The bytestring of the checkpoint. 
        """
        ss = time.perf_counter()
        bytes = BytesIO()
        torch.save(checkpoint, bytes)
        tt = time.perf_counter() - ss
        logging.info(f'Time elapsed saving checkpoint dict to bytes for {path}: {tt:.2f} seconds, rank {torch.distributed.get_rank()}')
        del checkpoint
        return bytes
    
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        if not S3Utils.is_s3_url(path):
            super().save_checkpoint(checkpoint, path, storage_options)
            return
 
        if os.path.exists(SHARED_MEM_DIR):
            localfile = self._serialize_checkpoint_to_shm(checkpoint, path)
            saved_as_file = True
        else:
            bytes = self._serialize_checkpoint_to_bytes(checkpoint, path)
            saved_as_file = False
 
        logging.info(f'Uploading checkpoint to {path} in synchronous mode, rank {torch.distributed.get_rank()}')
        if saved_as_file:
            _upload_file_to_s3(localfile, path, self.chunk_size_MB, self.max_write_concurrency, True)
        else:
            _upload_bytes_to_s3(bytes, path, self.chunk_size_MB, self.max_write_concurrency)
 
 
    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        if not S3Utils.is_s3_url(path):
            return super().load_checkpoint(path, map_location)
        
        if os.path.exists(SHARED_MEM_DIR):
            with NamedTemporaryFile(dir=SHARED_MEM_DIR, delete=True) as tempfile:
                logging.info(f'Loading checkpoint {path} into a temp file in shared memory {tempfile.name}, rank {torch.distributed.get_rank()}')
                S3Utils.download_s3_file_to_path(
                    s3_path=path, file_path=tempfile.name,
                    chunk_size_MB=self.chunk_size_MB,
                    max_concurrency=self.max_read_concurrency,
                )
                checkpoint = torch.load(tempfile.name)
        else:
            file_stream: BytesIO = S3Utils.download_s3_file_to_stream(
                s3_path=path,
                chunk_size_MB=self.chunk_size_MB,
                max_concurrency=self.max_read_concurrency
            )
            checkpoint = torch.load(file_stream)
        return checkpoint
    
    
    def remove_checkpoint(self, path: _PATH) -> None:
        if S3Utils.is_s3_url(path):
            S3Utils.remove_object(path)
        else:
            super().remove_checkpoint(path)
 
 
def _clean_up_conflicting_checkpoint(filepath: str) -> None:
    # before saving to s3, clean up any existing object with the same prefix megatron_gpt+step_count
    # e.g. before we save "megatron_gpt--step=1400-validation_loss=6.32-consumed_samples=55920.0-last.ckpt"
    # we need to clean up "megatron_gpt--step=1400-validation_loss=xxx-consumed_samples=yyy-last.ckpt"
    # so that in case later we need to resume from step 1400, it has a single checkpoint file at step 1400
    if S3Utils.is_s3_url(filepath):
        prefix_with_step = parse_prefix_with_step(filepath)
        logging.info(f'Cleaning up conflicting checkpoint under prefix {prefix_with_step}')
 
        conflict_last_ckpts = S3Utils.find_files_with_suffix(base_path=prefix_with_step, suffix='last.ckpt', return_key_only=False)
        logging.debug(f'Found last ckpts with same step value: {conflict_last_ckpts}')
        for last_ckpt in conflict_last_ckpts:
            logging.info(f'Cleaning up conflicting last ckpt {last_ckpt} before saving {filepath}')
            S3Utils.remove_object(last_ckpt)
 
def _upload_file_to_s3(localfile, path, chunk_size_MB, max_write_concurrency, remove_file):
    try :
        _clean_up_conflicting_checkpoint(path)
        S3Utils.upload_file_with_crt(localfile, path, chunk_size_MB, max_write_concurrency, remove_file)
    except Exception as e:
        logging.error(f'Failed to upload file {localfile} to {path} with exception {e}')
        raise e
 
def _upload_bytes_to_s3(bytes, path, chunk_size_MB, max_write_concurrency):
    try:
        _clean_up_conflicting_checkpoint(path)
        S3Utils.upload_file_stream_to_s3(bytes, path, chunk_size_MB, max_write_concurrency)
    except Exception as e:
        logging.error(f'Failed to upload bytes to {path} with exception {e}')
        raise e