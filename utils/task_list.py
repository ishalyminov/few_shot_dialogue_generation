from multiprocessing import Pool

tasks = []


def execute_tasks(in_callback, in_pool_size, initializer=None, initargs=[]):
    pool = Pool(processes=in_pool_size, initializer=initializer, initargs=initargs)
    try:
        result = pool.map(in_callback, tasks)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
    return result


def add_task(in_task):
    global tasks
    tasks.append(in_task)


def clear_task_list():
    global tasks
    tasks = []
