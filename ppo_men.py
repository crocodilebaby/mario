from memory_profiler import memory_usage
from stable_baselines3.common.callbacks import BaseCallback

class MemoryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MemoryCallback, self).__init__(verbose)
        self.step_counter = 0  # 添加一个计数器

    def _on_step(self):
        self.step_counter += 1
        if self.step_counter % 100 == 0:  # 每100步监控一次
            # 获取内存使用情况 (in MB)
            mem = memory_usage(-1, interval=1)[0]  # 移除了timeout参数
            self.logger.record('system/memory_usage_MB', mem)
        return True

