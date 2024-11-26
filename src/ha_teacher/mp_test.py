from multiprocessing import Process
import time


def worker():
    time.sleep(2)  # 模拟耗时任务


if __name__ == '__main__':
    num_processes = 10

    # 记录父进程启动前的时间
    start_time = time.time()

    processes = []
    for _ in range(num_processes):
        p = Process(target=worker)
        p.start()
        processes.append(p)

    # 等待所有子进程启动完成
    for p in processes:
        p.join()

    # 记录父进程启动后的时间
    end_time = time.time()

    print(f"启动 {num_processes} 个进程花费的时间: {end_time - start_time:.2f} 秒")
