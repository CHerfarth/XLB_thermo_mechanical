class BenchmarkData:
    _instance = None
    _wu = 0.0

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BenchmarkData, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @property
    def wu(self):
        return self._wu

    @wu.setter
    def wu(self, value):
        self._wu = value
