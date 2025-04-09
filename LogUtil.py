import os
import sys
import time
from datetime import datetime
import threading


class Log:
    def __init__(self, log_path, mode='a', level='INFO', verbose=True):
        """
        日志记录工具类

        参数：
        log_path: 日志文件路径
        mode: 文件打开模式 ('a' 追加模式 / 'w' 覆盖模式)
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        verbose: 是否在控制台显示日志
        """
        # 创建日志目录
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

        # 初始化配置
        self.log_file = open(log_path, mode)
        self.log_level = level.upper()
        self.verbose = verbose
        self.lock = threading.Lock()  # 线程安全锁
        self.level_map = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3
        }

    def __del__(self):
        """析构函数自动关闭文件"""
        self.close()

    def close(self):
        """手动关闭日志文件"""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def _write(self, level, message):
        """内部写入方法"""
        if self.level_map[level] < self.level_map[self.log_level]:
            return

        # 生成带时间戳的日志信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"

        # 线程安全写入
        with self.lock:
            try:
                if self.verbose:
                    print(log_msg, file=sys.stderr)
                self.log_file.write(log_msg + '\n')
                self.log_file.flush()  # 立即写入磁盘
            except IOError as e:
                sys.stderr.write(f"Log write failed: {str(e)}\n")

    # 不同日志级别的方法
    def debug(self, message):
        self._write('DEBUG', message)

    def info(self, message):
        self._write('INFO', message)

    def warning(self, message):
        self._write('WARNING', message)

    def error(self, message):
        self._write('ERROR', message)

    # 兼容原始调用的__call__方法
    def __call__(self, message, level='INFO'):
        """
        兼容原始代码的直接调用方式
        示例：log("训练开始", "INFO")
        """
        self._write(level.upper(), message)

    def set_level(self, level):
        """动态设置日志级别"""
        self.log_level = level.upper()

    def get_log_path(self):
        """获取日志文件路径"""
        return self.log_file.name
