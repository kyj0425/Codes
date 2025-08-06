from time import perf_counter


class Timer:
    time_stamps = []
    window_size = 100
    fps = 30

    @classmethod
    def set_time_stamp(cls):
        cls.time_stamps.append(perf_counter())
        if len(cls.time_stamps) > cls.window_size:
            del cls.time_stamps[0]
        cls.fps = cls.fps if len(cls.time_stamps) == 1 else (len(cls.time_stamps) - 1) / (cls.time_stamps[-1] - cls.time_stamps[0])

    @classmethod
    def get_fps(cls):
        return cls.fps
