from pathlib import Path
import pandas as pd
import logging

logger = logging.root
logFormatter = logging.Formatter('{relativeCreated:12.0f}ms {levelname:5s} [{filename}] {message:s}', style='{')
logger.setLevel(logging.DEBUG)


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Datasets
        self._train = None
        self._test = None
        self._sample = None


# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)


# ------------------ DOWNLOADS ------------------

    def data_as_png(
        self,
        name: str,
        *,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Returns a file as a pandas DataFrame.

        :param name: The name of the asked file. Must be a key in self.files.
        :param force_reload: A boolean indicating if we must force re-downloading the file.
        :returns: The pandas DataFrame corresponding to the asked file.
        """
        self.check_downloaded(name)
        return pd.read_csv(project.data / self.files[name]['save_as'], low_memory=False, sep=",")


project = CustomizedPath() 