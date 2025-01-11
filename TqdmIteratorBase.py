from abc import ABC, abstractmethod
from tqdm import tqdm


class TqdmIteratorBase(ABC):
    """
    A base class for creating iterators with a tqdm progress bar.

    Subclasses should implement the `generator` method to yield items.
    """

    def __init__(self, desc='Processing', total=None, **tqdm_kwargs):
        """
        Initializes the TqdmIteratorBase.

        :param desc: Description for the tqdm progress bar.
        :param total: Total number of iterations (optional).
        :param tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        self.desc = desc
        self.total = total
        self.tqdm_kwargs = tqdm_kwargs

    def __iter__(self):
        """
        Returns an iterator that wraps the generator with tqdm progress bar.
        """
        with tqdm(total=self.total, desc=self.desc, **self.tqdm_kwargs) as pbar:
            for item in self.generator():
                # Update the progress bar with each yielded item
                pbar.update(1)
                yield item

    @abstractmethod
    def generator(self):
        """
        Abstract method to be implemented by subclasses.
        Should yield items one by one.
        """
        pass
